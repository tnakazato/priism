from __future__ import annotations

import logging
from functools import partial
from collections import namedtuple
from collections.abc import Callable

import finufft
import numpy as np

from . import sparseimagingnufft

logger = logging.getLogger(__name__)

MAXITER = 50000
MINITER = 100
TD = 50
ETA = 1.1


# tuple holding MFISTA results
PyMfistaResults = namedtuple(
    'PyMfistaResults',
    [
        'M', 'N', 'NX', 'NY', 'N_active',
        'maxiter', 'ITER', 'nonneg',
        'lambda_l1', 'lambda_tsv', 'lambda_tv',
        'sq_error', 'mean_sq_error',
        'l1cost', 'tsvcost', 'tvcost',
        'finalcost', 'comp_time',
        'residual', 'Lip_const'
    ]
)


def calc_costs_nufft(
        M: int,
        Nx: int,
        Ny: int,
        u_dx: np.ndarray,
        v_dy: np.ndarray,
        vis_r: np.ndarray,
        vis_i: np.ndarray,
        vis_std: np.ndarray,
        lambda_l1: float,
        lambda_tv: float,
        lambda_tsv: float,
        nonneg: bool,
        xvec: np.ndarray,
        nthreads: int = 1,
        eps: float = 1e-6
) -> tuple[float, float, int, float, float]:
    if xvec.dtype not in (complex, np.complex64, np.complex128):
        _xvec = xvec.astype(complex)
    else:
        _xvec = xvec
    # isign=+1 matches the C++ engine's NUFFT2d2 convention (NU_SIGN=-1 in
    # mfista.hpp); finufft's default isign=-1 for type-2 produces the
    # complex-conjugate visibility relative to the C++ engine.
    model_vis = finufft.nufft2d2(
        u_dx,
        v_dy,
        _xvec,
        eps=eps,
        isign=+1,
        nthreads=nthreads
    )
    chisq: float = np.sum(
        (np.square(np.real(model_vis) - vis_r) + np.square(np.imag(model_vis) - vis_i))
        / np.square(vis_std)
    )
    l1cost: float = np.sum(np.abs(xvec))
    n_active: int = np.sum(np.where(xvec > 0, 1, 0))
    tsvcost: float = TSV(xvec)
    final_cost: float = chisq / 2 + lambda_l1 * l1cost + lambda_tsv * tsvcost
    return chisq, l1cost, n_active, tsvcost, final_cost


def soft_thresold(v: np.ndarray, eta: float) -> np.ndarray:
    """Apply soft-thresholding to an array.

    Args:
        v: input array
        eta: threshold value
    Returns:
        Soft-thresholded array
    """
    abs_eta = abs(eta)

    return np.sign(v) * np.maximum(np.abs(v) - abs_eta, 0.0)


def soft_thresold_nonneg(v: np.ndarray, eta: float) -> np.ndarray:
    """Apply soft-thresholding to a non-negative array.

    Args:
        v: input array
        eta: threshold value
    Returns:
        Soft-thresholded array
    """
    abs_eta = abs(eta)

    return np.maximum(v - abs_eta, 0.0)


def soft_threshold_box(vec: np.ndarray, eta: float, box_flag: bool, box: np.ndarray | None, threshold_func: Callable[[np.ndarray, float], np.ndarray]) -> np.ndarray:
    """Apply soft-thresholding with box constraints.

    Args:
        vec: input numpy array
        eta: threshold value
        box_flag: if True, apply box constraints
        box: box constraints array (same shape as vec) or None
        threshold_func: vectorized function to apply for thresholding

    Returns:
        Soft-thresholded numpy array
    """
    nvec = threshold_func(vec, eta)

    if box_flag:
        nvec = nvec * box

    return nvec


def TSV(xvec: np.ndarray) -> float:
    """Compute the Total Squared Variation (TSV) of an image.

    Args:
        xvec: 2D numpy array representing the image (length Nx*Ny)

    Returns:
        TSV value as a float
    """
    return np.square(np.diff(xvec, axis=0)).sum() + \
        np.square(np.diff(xvec, axis=1)).sum()


def d_TSV(xvec: np.ndarray) -> np.ndarray:
    """Compute the gradient of the Total Squared Variation (TSV) of an image.

    Args:
        xvec: 2D numpy array representing the image (length Nx*Ny)

    Returns:
        Gradient of TSV as a numpy array of the same shape as xvec
    """
    dtmp = np.zeros(xvec.shape, dtype=float)

    # Vertical differences
    diff_x = np.diff(xvec, axis=0)
    dtmp[:-1, :] -= 2 * diff_x
    dtmp[1:, :] += 2 * diff_x

    # Horizontal differences
    diff_y = np.diff(xvec, axis=1)
    dtmp[:, :-1] -= 2 * diff_y
    dtmp[:, 1:] += 2 * diff_y

    return dtmp


def calc_Q_part(xvec1: np.ndarray, xvec2: np.ndarray, c: float, AyAz: np.ndarray) -> float:
    """Calculate the Q part of the cost function.

    Args:
        xvec1: first image array (length Nx*Ny)
        xvec2: second image array (length Nx*Ny)
        c: current c parameter
        AyAz: auxiliary array (length Nx*Ny)

    Returns:
        Q part value as a float
    """
    # x1 - x2
    buf_vec = xvec1 - xvec2
    # (x1 - x2)'A'(y - A x2)
    term1 = np.dot(buf_vec, AyAz)
    # (x1 - x2)'(x1 - x2)
    term2 = np.square(buf_vec).sum()

    logger.debug("term1: %s", term1)
    logger.debug("term2: %s", term2)

    return -term1 + c * term2 / 2


def x2y_nufft(
        u: np.ndarray,
        v: np.ndarray,
        xvec: np.ndarray,
        nthreads: int = 1,
        eps: float = 1e-6
    ) -> np.ndarray:
    """Forward-only NUFFT: model visibility from an image, with no residual
    or weighting applied. u/v must already be in the NUFFT radian convention
    (see sparseimagingnufft.SparseImagingInputsNUFFT.convert_uv), matching
    calc_F_part_nufft's internal call to finufft.nufft2d2 below.

    Used by external self-calibration code (priism-selfcal) to compute the
    model visibility for the current image estimate, mirroring the C++
    engine's x2y_nufft.

    Args:
        u: u coordinates of visibilities (length M), NUFFT radian convention
        v: v coordinates of visibilities (length M), NUFFT radian convention
        xvec: model image array (shape (Nx, Ny))
        nthreads: number of threads finufft may use
        eps: precision required for NUFFT (default: 1e-6)

    Returns:
        model_vis: model visibility array (length M)
    """
    if xvec.dtype not in (complex, np.complex64, np.complex128):
        _xvec = xvec.astype(complex)
    else:
        _xvec = xvec
    return finufft.nufft2d2(u, v, _xvec, eps=eps, isign=+1, nthreads=nthreads)


def calc_F_part_nufft(
        u: np.ndarray,
        v: np.ndarray,
        vis: np.ndarray,
        weight: np.ndarray,
        xvec: np.ndarray,
        nthreads: int = 1,
        eps: float = 1e-6
    ) -> np.ndarray:
    """Calculate the F part of the cost function using NUFFT.

    Args:
        u: u coordinates of visibilities (length M)
        v: v coordinates of visibilities (length M)
        vis: observed visibilities (length M)
        weight: visibility weights (length M)
        xvec: model image array (shape (Nx, Ny))
        nthreads: number of threads finufft may use
        eps: precision required for NUFFT (default: 1e-6)

    Returns:
        yAx: visibility difference array (length M)
    """
    if xvec.dtype not in (complex, np.complex64, np.complex128):
        _xvec = xvec.astype(complex)
    else:
        _xvec = xvec
    # isign=+1 matches the C++ engine's NUFFT2d2 convention, see calc_costs_nufft
    model_vis = finufft.nufft2d2(
        u,
        v,
        _xvec,
        eps=eps,
        isign=+1,
        nthreads=nthreads
    )
    logger.debug(
        "model_vis = %s, %s ~ %s, %s",
        model_vis.imag.min(), model_vis.real.min(), model_vis.imag.max(), model_vis.real.max()
    )
    yAx = (vis - model_vis) * weight
    logger.debug(
        "yAx = %s, %s ~ %s, %s",
        yAx.imag.min(), yAx.real.min(), yAx.imag.max(), yAx.real.max()
    )

    return yAx


def dF_dx_nufft(
        u: np.ndarray,
        v: np.ndarray,
        weight: np.ndarray,
        yAx: np.ndarray,
        nx: int,
        ny: int,
        nthreads: int = 1,
        eps: float = 1e-6
    ) -> np.ndarray:
    """Compute the gradient of F part using NUFFT.

    Args:
        u: u coordinates of visibilities (length M)
        v: v coordinates of visibilities (length M)
        weight: visibility weights (length M)
        yAx: visibility difference array (length M)
        nx: image x dimension
        ny: image y dimension
        nthreads: number of threads finufft may use
        eps: precision required for NUFFT (default: 1e-6)

    Returns:
        dfdx: gradient array in image domain (length Nx*Ny)
    """
    # TODO: not add complex conjugate to the visibility
    weighted_yAx = yAx * weight
    # yAx containing complex conjugate
    u_full = np.empty(len(u) * 2, dtype=float)
    v_full = np.empty(len(v) * 2, dtype=float)
    weighted_yAx_full = np.empty(len(weighted_yAx) * 2, dtype=complex)
    u_full[:len(u)] = u
    u_full[len(u):] = -u
    v_full[:len(v)] = v
    v_full[len(v):] = -v
    weighted_yAx_full[:len(weighted_yAx)] = weighted_yAx
    weighted_yAx_full[len(weighted_yAx):] = np.conjugate(weighted_yAx)
    # isign=-1 here pairs as the true adjoint of the isign=+1 forward
    # transform in calc_F_part_nufft, matching the C++ engine's gradient
    dfdx = finufft.nufft2d1(
        u_full,
        v_full,
        weighted_yAx_full,
        eps=eps,
        isign=-1,
        n_modes=(nx, ny),
        nthreads=nthreads
    ) / 2
    logger.debug("dfdx.imag min: %s, dfdx.real min: %s", dfdx.imag.min(), dfdx.real.min())
    logger.debug("dfdx.imag max: %s, dfdx.real max: %s", dfdx.imag.max(), dfdx.real.max())
    return dfdx.real


def mfista_L1_TSV_core_nufft(
        M: int,
        Nx: int,
        Ny: int,
        u_dx: np.ndarray,
        v_dy: np.ndarray,
        maxiter: int,
        eps: float,
        vis_r: np.ndarray,
        vis_i: np.ndarray,
        vis_std: np.ndarray,
        lambda_l1: float,
        lambda_tsv: float,
        cinit: float,
        xinit: np.ndarray,
        nonneg_flag: bool,
        box_flag: bool,
        cl_box: np.ndarray,
        restart_flag: bool = True,
        nthreads: int = 1
) -> tuple[PyMfistaResults, np.ndarray]:
    """Python translation of the C++ mfista_L1_TSV_core_nufft.

    Args:
        M: number of visibilities
        Nx: image x dimension
        Ny: image y dimension
        u_dx: array of visibility u coordinates (length M)
        v_dy: array of visibility v coordinates (length M)
        maxiter: maximum number of iterations
        eps: stopping threshold
        vis_r: real part of observed visibilities (length M)
        vis_i: imaginary part of observed visibilities (length M)
        vis_std: visibility sigma (length M)
        lambda_l1: L1 regularization parameter
        lambda_tsv: TSV regularization parameter
        cinit: initial c parameter
        xinit: initial image array (length Nx*Ny)
        nonneg_flag: if True, enforce non-negativity
        box_flag: if True, enable clean box
        cl_box: clean box
        restart_flag: if True, apply Nesterov gradient restart (O'Donoghue
            & Candes 2015) to the momentum coefficient instead of letting
            it grow monotonically every iteration
        nthreads: number of threads finufft may use per NUFFT call. Default
            is 1 to avoid oversubscribing when multiple solves run
            concurrently (e.g. cross-validation grid search); pass a higher
            value to speed up a single solve if you know it is not sharing
            the machine with other parallel work.

    Returns:
        Result tuple (PyMfistaResults) and final image array
    """

    NN = Nx * Ny
    imshape = (Nx, Ny)

    # Map inputs to numpy arrays / copies where necessary
    u = np.asarray(u_dx, dtype=float)
    v = np.asarray(v_dy, dtype=float)
    vis = np.asarray(
        [complex(r, i) for r, i in zip(vis_r, vis_i)],
        dtype=complex
    )
    weight = np.reciprocal(vis_std, dtype=float)

    # allocate arrays corresponding to the C++ Eigen vectors
    cost = np.zeros(maxiter, dtype=float)
    xnew = np.zeros(NN, dtype=float)
    xtmp = np.zeros(NN, dtype=float)
    dtmp = np.zeros(NN, dtype=float)
    box = np.zeros(NN, dtype=float)

    # copy initial x
    # shape is (NN,)
    xvec = np.asarray(xinit, dtype=float).copy()
    zvec = xvec.copy()

    if box_flag == 1:
        box = np.asarray(cl_box, dtype=float).copy()
    else:
        box = np.zeros(NN, dtype=float)

    logger.debug("Memory allocation and preparations.")

    logger.debug("Preparation for FFT.")

    # placeholders for FFTW plans
    # (C++ creates and executes plans;
    # leave as None/placeholders)
    fftwplan_c2r = None
    fftwplan_r2c = None

    # choose soft-threshold function
    if nonneg_flag == 0:
        soft_th_box = partial(soft_threshold_box, threshold_func=soft_thresold)
    elif nonneg_flag == 1:
        soft_th_box = partial(soft_threshold_box, threshold_func=soft_thresold_nonneg)
    else:
        logger.error("nonneg_flag must be chosen properly.")
        return 0, 0.0, np.array([])

    logger.debug("Done.")

    c = cinit

    logger.debug("Computing image with MFISTA using NUFFT.")
    logger.debug("Stop if iter = %d or Delta_cost < %s", maxiter, eps)

    # initial cost
    # Fourier transformation of xvec (model image) using NUFFT
    # input:
    #   xvec: model image
    # output:
    #   yAx: visibility diff, (model - observed) * weight
    #   return value: Chi-square term (1/2 * norm of yAx)
    yAx = calc_F_part_nufft(u, v, vis, weight, xvec.reshape(imshape), nthreads=nthreads, eps=eps)
    costtmp = np.square(np.abs(yAx)).sum() / 2
    logger.debug("costtmp (initial): %s", costtmp)
    # looks like equivalent is
    #   1. perform NUFFT to get model visibilities from xvec
    #   2. compute (model - observed) * weight
    #   3. compute 0.5 * norm squared of the result

    # add L1 normalizaton term to the cost
    l1cost = np.sum(np.abs(xvec))
    costtmp += lambda_l1 * l1cost
    logger.debug("costtmp (L1): %s", costtmp)

    # add TSV term to the cost
    if lambda_tsv > 0:
        tsvcost = TSV(xvec.reshape(imshape))
        costtmp += lambda_tsv * tsvcost
        logger.debug("costtmp (TSV): %s", costtmp)

    eta = 10.0
    iter_cnt = 0
    mu = 1.0

    for iter_cnt in range(maxiter):
        cost[iter_cnt] = costtmp

        logger.debug("%5d cost = %.5f, c = %s", iter_cnt + 1, cost[iter_cnt], c)

        # compute Chi-square part
        # input:
        #   zvec: current image (initially same as xvec == xinit)
        # output:
        #   yAx: visibility diff, (model - observed) * weight
        #   return value: Chi-square term (1/2 * norm of yAx)
        yAx = calc_F_part_nufft(u, v, vis, weight, zvec.reshape(imshape), nthreads=nthreads)
        Qcore = np.square(np.abs(yAx)).sum() / 2
        logger.debug("Qcore: %s", Qcore)

        # compute gradient dF/dx at zvec
        # input:
        #   yAx: visibility diff, (model - observed) * weight
        # output:
        #   dfdx: Fourier transform of visibility gradient, (model - observed) * weight**2
        #         shape is (Nx, Ny)
        dfdx = dF_dx_nufft(u, v, weight, yAx, Nx, Ny, nthreads=nthreads, eps=eps)
        # looks like equivalent is
        #   1. compute visibility gradient: (model - observed) * weight**2
        #   1. perform adjoint NUFFT to get image domain gradient from yAx

        if lambda_tsv > 0.0:
            # add TSV term to the cost
            tsvcost = TSV(zvec.reshape(imshape))
            Qcore += lambda_tsv * tsvcost

            # compute gradient of TSV term
            dtmp = d_TSV(zvec.reshape(imshape))
            dfdx = dfdx - lambda_tsv * dtmp

        # inner loop for line-search / backtracking
        Fval = 0.0
        for i in range(maxiter):
            logger.debug("Backtracking: iter %d", i)
            xtmp = zvec + dfdx.ravel() / c
            logger.debug("    zvec = %s ~ %s", zvec.min(), zvec.max())
            logger.debug("    xtmp = %s ~ %s", xtmp.min(), xtmp.max())
            # no box_flag
            #   xnew = max(xtmp - lambda_l1 / c, 0)
            # with box_flag
            #   xnew = max(xtmp - lambda_l1 / c, 0) if box > 0 else 0
            xnew = soft_th_box(xtmp, lambda_l1 / c, box_flag, box)
            logger.debug("    xnew = %s ~ %s", xnew.min(), xnew.max())

            # compute cost at xnew
            yAx = calc_F_part_nufft(u, v, vis, weight, xnew.reshape(imshape), nthreads=nthreads)
            Fval = np.square(np.abs(yAx)).sum() / 2

            if lambda_tsv > 0.0:
                # add TSV term to the cost
                tsvcost = TSV(xnew.reshape(imshape))
                Fval += lambda_tsv * tsvcost

            Qval = calc_Q_part(xnew, zvec, c, dfdx.ravel())
            logger.debug("Qcore: %s, Qval: %s", Qcore, Qval)
            Qval += Qcore

            logger.debug("Fval %s Qval %s", Fval, Qval)
            if Fval <= Qval:
                break

            c *= eta

        logger.debug("iter %d, xnew mean %s, std %s", iter_cnt, xnew.mean(), xnew.std())

        eta = ETA  # keep same ETA constant name as in C++ context
        c /= eta

        munew = (1.0 + np.sqrt(1.0 + 4.0 * mu * mu)) / 2.0

        l1cost = np.sum(np.abs(xnew))
        Fval += lambda_l1 * l1cost

        # matches C++ `zvec = xvec;` here: the momentum update below must
        # combine xnew with the *previous xvec*, not the previous zvec
        y_k = zvec.copy()
        z_old = xvec.copy()

        if Fval < cost[iter_cnt]:
            costtmp = Fval

            tmpa = 1.0 + ((mu - 1.0) / munew)
            tmpb = ((1.0 - mu) / munew)

            zvec = tmpa * xnew + tmpb * z_old
            xvec = xnew.copy()
        else:
            tmpa = mu / munew
            tmpb = 1.0 - (mu / munew)
            zvec = tmpa * xnew + tmpb * z_old

            # another stopping rule
            if (iter_cnt > 1) and (np.sum(np.abs(xvec)) == 0.0):
                logger.info("stop iteration because xvec is all zero")
                break

        # stopping condition from C++ (uses MINITER and TD)
        if (iter_cnt >= MINITER) \
           and (cost[iter_cnt - TD] - cost[iter_cnt] < eps):
            logger.info(f"converged at iter {iter_cnt}")
            break

        if restart_flag:
            # Gradient restart (O'Donoghue & Candes 2015): function restart
            # (F(x_k) > F(x_{k-1})) never fires under MFISTA's monotone
            # accept/reject rule, so it is not useful here. Instead, detect
            # when the momentum point y_k = zvec is pointing the "wrong way"
            # relative to the actual step just taken (xnew - z_old); when it
            # is, forget the accumulated momentum by resetting mu to 1
            # instead of letting it keep growing via mu = munew.
            if np.dot(y_k - xnew, xnew - z_old) > 0.0:
                mu = 1.0
            else:
                mu = munew
        else:
            mu = munew

    # end main loop

    # adjust final iter value same as C++ behavior
    if iter_cnt + 1 == maxiter:
        logger.debug(f"{iter_cnt:5d} cost = {cost[iter_cnt-1]:.5f}")
        iter_out = iter_cnt
    else:
        logger.debug(f"{iter_cnt+1:5d} cost = {cost[iter_cnt]:.5f}")
        iter_out = iter_cnt

    # update cinit if mutable container provided
    # if (hasattr(cinit, '__len__') and len(cinit) > 0):
    #     try:
    #         cinit[0] = c
    #     except Exception:
    #         # if it's a numpy scalar array
    #         try:
    #             cinit[0] = c
    #         except Exception:
    #             pass

    # return iter_out + 1, c, xvec
    chisq, l1cost, n_active, tsvcost, final_cost = calc_costs_nufft(
        M=M,
        Nx=Nx,
        Ny=Ny,
        u_dx=u,
        v_dy=v,
        vis_r=vis_r,
        vis_i=vis_i,
        vis_std=vis_std,
        lambda_l1=lambda_l1,
        lambda_tv=0.0,
        lambda_tsv=lambda_tsv,
        nonneg=nonneg_flag,
        xvec=xvec.reshape(imshape),
        nthreads=nthreads,
        eps=eps
    )
    result = PyMfistaResults(
            M=M,
            N=Nx * Ny,
            NX=Nx,
            NY=Ny,
            N_active=n_active,
            maxiter=maxiter,
            ITER=iter_out + 1,
            nonneg=nonneg_flag,
            lambda_l1=lambda_l1,
            lambda_tsv=lambda_tsv,
            lambda_tv=0,
            sq_error=chisq,
            mean_sq_error=chisq / M,
            l1cost=l1cost,
            tsvcost=tsvcost,
            tvcost=0.0,
            finalcost=final_cost,
            comp_time=0.0,  # to be computed later if needed
            residual=None,  # to be computed later if needed
            Lip_const=c
        )
    return result, xvec



PySparseImagingInputs = sparseimagingnufft.SparseImagingInputsNUFFT


class PySparseImagingResults:
    def __init__(self, nx, ny, initialimage=None):
        self.nx = nx
        self.ny = ny
        nn = self.nx * self.ny
        self.xinit = np.empty(nn, dtype=float)
        if initialimage is None:
            self.xinit[:] = 0
        else:
            self.xinit[:] = initialimage
        self.xout = np.empty([], dtype=float)
        self.mfista_result: PyMfistaResults | None = None

    @property
    def image(self):
        img = self.xout.reshape((self.nx, self.ny))
        return img


class SparseImagingExecutor:
    Inputs = PySparseImagingInputs

    def __init__(self, lambda_L1: float, lambda_TV: float = 0.0, lambda_TSV: float = 0.0,
                 cinit: float = 5e10, nonnegative: bool = True):
        self.lambda_L1 = lambda_L1
        self.lambda_TV = lambda_TV
        self.lambda_TSV = lambda_TSV
        self.cinit = cinit
        self.nonnegative = nonnegative

        self.outfile = 'x.out'

    def run(self, inputs: PySparseImagingInputs, initialimage: np.ndarray | None = None,
            maxiter: int = 50000, eps: float = 1.0e-5, cl_box: np.ndarray | None = None,
            nthreads: int = 1):
        """
        Run MFISTA routine to get an image

        nthreads -- number of threads finufft may use per NUFFT call.
                    Default is 1 (see mfista_L1_TSV_core_nufft for rationale).
        """
        # input summary
        logger.debug(f'lambda_l1 = {self.lambda_L1}')
        logger.debug(f'lambda_tv = {self.lambda_TV}')
        logger.debug(f'lambda_tsv = {self.lambda_TSV}')
        logger.debug(f'c = {self.cinit:g}')
        logger.debug('')
        logger.debug(f'number of u-v points: {inputs.m}')
        logger.debug(f'X-dim of image:       {inputs.nx}')
        logger.debug(f'Y-dim of image:       {inputs.ny}')

        # run MFISTA
        result = PySparseImagingResults(inputs.nx, inputs.ny, initialimage=initialimage)
        chisq_initial, l1cost_initial, n_active_initial, tsvcost_initial, final_cost_initial = calc_costs_nufft(
            M=inputs.m,
            Nx=inputs.nx,
            Ny=inputs.ny,
            u_dx=inputs.u,
            v_dy=inputs.v,
            vis_r=inputs.yreal,
            vis_i=inputs.yimag,
            vis_std=inputs.noise,
            lambda_l1=self.lambda_L1,
            lambda_tv=0.0,
            lambda_tsv=self.lambda_TSV,
            nonneg=self.nonnegative,
            xvec=result.xinit.reshape((inputs.nx, inputs.ny)),
            nthreads=nthreads
        )

        mfista_result, xout = mfista_L1_TSV_core_nufft(
            M=inputs.m,
            Nx=inputs.nx,
            Ny=inputs.ny,
            u_dx=inputs.u,
            v_dy=inputs.v,
            maxiter=maxiter,
            eps=eps,
            vis_r=inputs.yreal,
            vis_i=inputs.yimag,
            vis_std=inputs.noise,
            lambda_l1=self.lambda_L1,
            lambda_tsv=self.lambda_TSV,
            cinit=self.cinit,
            xinit=result.xinit,
            nonneg_flag=self.nonnegative,
            box_flag=cl_box is not None,
            cl_box=cl_box if cl_box is not None else np.array([]),
            nthreads=nthreads,
        )

        result.mfista_result = mfista_result
        result.xout = xout

        # show IO filenames
        self._show_io_info(inputs, initialimage)

        # show result
        self._show_result(result.mfista_result)

        return result

    def _show_io_info(self, inputs: PySparseImagingInputs, initialimage: np.ndarray | None):
        # show IO filenames
        logger.debug('\n\nInput/Output summary:\n')
        logger.debug(f' Input u-v data file:      {inputs.infile}')
        if initialimage is None:
            logger.debug(' x was initialized with 0.0')
        else:
            logger.debug(' x was initialize by the user\n')

    def _show_result(self, mfista_result):
        # show results
        logger.info('')
        logger.info('')
        logger.info("Size of the problem:")
        logger.info(f' size of input vector:  {mfista_result.M}')
        logger.info(f' size of output vector: {mfista_result.N}')
        if mfista_result.NX != 0:
            logger.info(f'size of image:          {mfista_result.NX} x {mfista_result.NY}')

        logger.info("")
        logger.info("Problem Setting:")
        if mfista_result.nonneg == 1:
            logger.info(' x is a nonnegative vector.\n')
        elif mfista_result.nonneg == 0:
            logger.info(' x is a real vector (takes 0, positive, and negative value).\n')

        if mfista_result.lambda_l1 != 0:
            logger.info(f' Lambda_l1: {mfista_result.lambda_l1:e}')
        if mfista_result.lambda_tsv != 0:
            logger.info(f' Lambda_tsv: {mfista_result.lambda_tsv:e}')
        if mfista_result.lambda_tv != 0:
            logger.info(f' Lambda_tv: {mfista_result.lambda_tv:e}')
        logger.info(f' MAXITER: {mfista_result.maxiter}')

        logger.info("")
        logger.info('Results:')
        logger.info(f' # of iterations:       {mfista_result.ITER}')
        logger.info(f' cost:                  {mfista_result.finalcost:e}')
        logger.info(f' computation time[sec]: {mfista_result.comp_time:e}\n')
        logger.info(f' # of nonzero pixels:   {mfista_result.N_active}')
        logger.info(f' Squared Error (SE):    {mfista_result.sq_error:e}')
        logger.info(f' Mean SE:               {mfista_result.mean_sq_error:e}')
        if mfista_result.lambda_l1 != 0:
            logger.info(f' L1 cost:               {mfista_result.l1cost:e}')
        if mfista_result.lambda_tsv != 0:
            logger.info(f' TSV cost:              {mfista_result.tsvcost:e}')
        if mfista_result.lambda_tv != 0:
            logger.info(f' TV cost:               {mfista_result.tvcost:e}')
        logger.debug('LOOE:    Could not be computed because Hessian was not positive definite.')
