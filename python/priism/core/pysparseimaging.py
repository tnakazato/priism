from __future__ import annotations

from functools import partial
from collections import namedtuple
from collections.abc import Callable

import finufft
import numpy as np

from . import sparseimagingnufft


# MSP = 6
# MSP2 = MSP * 2
# MSP4 = MSP * 4
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
        xvec: np.ndarray
) -> tuple[float, float, int, float, float]:
    if xvec.dtype not in (complex, np.complex64, np.complex128):
        _xvec = xvec.astype(complex)
    else:
        _xvec = xvec
    # isign=+1 matches the C++ engine's NUFFT2d2 convention (NU_SIGN=-1 in
    # mfista.hpp); finufft's default isign=-1 for type-2 produces the
    # complex-conjugate visibility relative to the C++ engine.
    model_vis = finufft.nufft2d2(u_dx, v_dy, _xvec, eps=1e-12, isign=+1) # / len(u_dx)
    chisq: float = np.sum(
        (np.square(np.real(model_vis) - vis_r) + np.square(np.imag(model_vis) - vis_i))
        / np.square(vis_std)
    )
    l1cost: float = np.sum(np.abs(xvec))
    n_active: int = np.sum(np.where(xvec > 0, 1, 0))
    tsvcost: float = TSV(xvec)
    final_cost: float = chisq / 2 + lambda_l1 * l1cost + lambda_tsv * tsvcost
    # results = PyMfistaResults(
    #     M=M,
    #     N=Nx * Ny,
    #     NX=Nx,
    #     NY=Ny,
    #     maxiter=0,
    #     ITER=0,
    #     nonneg=0,
    #     lambda_l1=lambda_l1,
    #     lambda_tsv=lambda_tsv,
    #     lambda_tv=lambda_tv,
    #     sq_error=chisq,
    #     mean_sq_error=chisq / M,
    #     l1cost=l1cost,
    #     tsvcost=tsvcost,
    #     tvcost=0,
    #     N_active=n_active,
    #     finalcost=chisq / 2 + lambda_l1 * l1cost + lambda_tsv * tsvcost,
    #     comp_time=0.0,
    #     residual=0.0,
    #     Lip_const=0.0
    # )
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

    print(f"term1: {term1}")
    print(f"term2: {term2}")

    return -term1 + c * term2 / 2


def calc_F_part_nufft(u: np.ndarray, v: np.ndarray, vis: np.ndarray, weight: np.ndarray, xvec: np.ndarray) -> np.ndarray:
    """Calculate the F part of the cost function using NUFFT.

    Args:
        u: u coordinates of visibilities (length M)
        v: v coordinates of visibilities (length M)
        vis: observed visibilities (length M)
        weight: visibility weights (length M)
        xvec: model image array (shape (Nx, Ny))

    Returns:
        yAx: visibility difference array (length M)
    """
    if xvec.dtype not in (complex, np.complex64, np.complex128):
        _xvec = xvec.astype(complex)
    else:
        _xvec = xvec
    # isign=+1 matches the C++ engine's NUFFT2d2 convention, see calc_costs_nufft
    model_vis = finufft.nufft2d2(u, v, _xvec, eps=1e-12, isign=+1) # / len(u)
    print(f"model_vis = {model_vis.imag.min()}, {model_vis.real.min()} ~ {model_vis.imag.max()}, {model_vis.real.max()}")
    yAx = (vis - model_vis) * weight
    print(f"yAx = {yAx.imag.min()}, {yAx.real.min()} ~ {yAx.imag.max()}, {yAx.real.max()}")

    return yAx


def dF_dx_nufft(u: np.ndarray, v: np.ndarray, weight: np.ndarray, yAx: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """Compute the gradient of F part using NUFFT.

    Args:
        u: u coordinates of visibilities (length M)
        v: v coordinates of visibilities (length M)
        weight: visibility weights (length M)
        yAx: visibility difference array (length M)
        nx: image x dimension
        ny: image y dimension

    Returns:
        dfdx: gradient array in image domain (length Nx*Ny)
    """
    # TODO: add complex conjugate to the visibility
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
    dfdx = finufft.nufft2d1(u_full, v_full, weighted_yAx_full, eps=1e-12, isign=-1, n_modes=(nx, ny)) / 2
    # dfdx = finufft.nufft2d1(u, v, weighted_yAx, eps=1e-12, n_modes=(nx, ny))
    print(f"dfdx.imag min: {dfdx.imag.min()}, dfdx.real min: {dfdx.real.min()}")
    print(f"dfdx.imag max: {dfdx.imag.max()}, dfdx.real max: {dfdx.real.max()}")
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
        cl_box: np.ndarray
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
    # rvec = np.zeros(4 * NN, dtype=float)
    cost = np.zeros(maxiter, dtype=float)
    # dfdx = np.zeros(NN, dtype=float)
    xnew = np.zeros(NN, dtype=float)
    xtmp = np.zeros(NN, dtype=float)
    dtmp = np.zeros(NN, dtype=float)
    box = np.zeros(NN, dtype=float)

    # complex arrays
    # yAx = np.zeros(M, dtype=complex)
    # cvec = np.zeros(2 * Nx * (Ny + 1), dtype=complex)
    # buf_ax = np.zeros(2 * Nx, dtype=complex)

    # E arrays and index arrays (placeholders -- preNUFFT will fill them)
    # mx = np.zeros(M, dtype=int)
    # my = np.zeros(M, dtype=int)
    # y_neg = np.zeros(M, dtype=int)
    # E1 = np.zeros(M, dtype=float)
    # E2x = np.zeros((MSP2, M), dtype=float)
    # E2y = np.zeros((MSP2, M), dtype=float)
    # E4 = np.zeros(NN, dtype=float)

    # mbuf_l = np.zeros((2 * Nx + MSP2, Ny + MSP2 + 1), dtype=complex)
    # mbuf_h = np.zeros((Ny + MSP2 + 1, 2 * Nx + MSP2), dtype=complex)

    # copy initial x
    # shape is (NN,)
    xvec = np.asarray(xinit, dtype=float).copy()
    zvec = xvec.copy()

    if box_flag == 1:
        box = np.asarray(cl_box, dtype=float).copy()
    else:
        box = np.zeros(NN, dtype=float)

    # sort input (kept as C++ name, expects Python binding)
    # sort_input(
    #     M,
    #     Nx,
    #     Ny,
    #     u,
    #     v,
    #     vis.real,
    #     vis.imag,
    #     vis_std
    # )

    print("Memory allocation and preparations.\n")

    # tile boundary container
    # tile_boundary = []

    print("Preparation for FFT.")

    # prepare for nufft (keeps same call)
    # preNUFFT: TBD
    # fills E1, E2x, E2y, E4, mx, my, y_neg
    # preNUFFT(M, Nx, Ny, u, v, E1, E2x, E2y, E4, mx, my, y_neg)

    # # configure tiles (use available CPU count)
    # try:
    #     nthreads = multiprocessing.cpu_count()
    # except Exception:
    #     nthreads = 1
    # # call configure_tile as in C++ (expects mbuf_l.cols() equivalent)
    # configure_tile(mbuf_l.shape[1], nthreads, Ny, my, tile_boundary)

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
        print("nonneg_flag must be chosen properly.")
        return 0, 0.0, np.array([])

    print(" Done.\n")

    c = cinit

    print("Computing image with MFISTA using NUFFT.")
    print(f"Stop if iter = {maxiter} or Delta_cost < {eps}\n")

    # initial cost
    # Fourier transformation of xvec (model image) using NUFFT
    # input:
    #   xvec: model image
    # output:
    #   yAx: visibility diff, (model - observed) * weight
    #   return value: Chi-square term (1/2 * norm of yAx)
    # costtmp = calc_F_part_nufft(
    #     M, Nx, Ny, yAx, E1, E2x, E2y, E4, mx, my, y_neg,
    #     rvec, cvec, fftwplan_r2c, vis, weight, xvec, mbuf_h
    # )
    print(f"len(u) = {len(u)}, len(v) = {len(v)}, abs(vis).min = {np.abs(vis).min()}, abs(vis).max = {np.abs(vis).max()}, weight.min = {weight.min()}, weight.max = {weight.max()}")
    print(f"u = {u.min()} ~ {u.max()}")
    print(f"v = {v.min()} ~ {v.max()}")
    yAx = calc_F_part_nufft(u, v, vis, weight, xvec.reshape(imshape))
    print(f"yAx shape: {yAx.shape}")
    costtmp = np.square(np.abs(yAx)).sum() / 2
    print(f"costtmp (initial): {costtmp}")

    # looks like equivalent is
    #   1. perform NUFFT to get model visibilities from xvec
    #   2. compute (model - observed) * weight
    #   3. compute 0.5 * norm squared of the result

    # add L1 normalizaton term to the cost
    l1cost = np.sum(np.abs(xvec))
    costtmp += lambda_l1 * l1cost
    print(f"costtmp (L1): {costtmp}")

    # add TSV term to the cost
    if lambda_tsv > 0:
        tsvcost = TSV(xvec.reshape(imshape))
        costtmp += lambda_tsv * tsvcost
        print(f"costtmp (TSV): {costtmp}")

    eta = 10.0
    iter_cnt = 0
    mu = 1.0

    for iter_cnt in range(maxiter):
        cost[iter_cnt] = costtmp

        # TODO: remove True
        if True or (iter_cnt % 10) == 0:
            print(f"{iter_cnt+1:5d} cost = {cost[iter_cnt]:.5f}, c = {c}")

        # compute Chi-square part
        # input:
        #   zvec: current image (initially same as xvec == xinit)
        # output:
        #   yAx: visibility diff, (model - observed) * weight
        #   return value: Chi-square term (1/2 * norm of yAx)
        # Qcore = calc_F_part_nufft(
        #     M, Nx, Ny, yAx, E1, E2x, E2y, E4, mx, my, y_neg,
        #     rvec, cvec, fftwplan_r2c, vis, weight, zvec, mbuf_h
        # )
        yAx = calc_F_part_nufft(u, v, vis, weight, zvec.reshape(imshape))
        Qcore = np.square(np.abs(yAx)).sum() / 2
        print(f"Qcore: {Qcore}")

        # compute gradient dF/dx at zvec
        # input:
        #   yAx: visibility diff, (model - observed) * weight
        # output:
        #   dfdx: Fourier transform of visibility gradient, (model - observed) * weight**2
        #         shape is (Nx, Ny)
        # dF_dx_nufft(
        #     M, Nx, Ny, dfdx, E1, E2x, E2y, E4, mx, my, y_neg, buf_ax,
        #     cvec, rvec, fftwplan_c2r, weight, yAx, mbuf_l, tile_boundary
        # )
        dfdx = dF_dx_nufft(u, v, weight, yAx, Nx, Ny)
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
            print(f"Backtracking: iter {i}")
            xtmp = zvec + dfdx.ravel() / c
            print(f"    zvec = {zvec.min()} ~ {zvec.max()}")
            print(f"    xtmp = {xtmp.min()} ~ {xtmp.max()}")
            # no box_flag
            #   xnew = max(xtmp - lambda_l1 / c, 0)
            # with box_flag
            #   xnew = max(xtmp - lambda_l1 / c, 0) if box > 0 else 0
            xnew = soft_th_box(xtmp, lambda_l1 / c, box_flag, box)
            print(f"    xnew = {xnew.min()} ~ {xnew.max()}")

            # compute cost at xnew
            # Fval = calc_F_part_nufft(
            #     M, Nx, Ny, yAx, E1, E2x, E2y, E4, mx, my, y_neg,
            #     rvec, cvec, fftwplan_r2c, vis, weight, xnew, mbuf_h
            # )
            yAx = calc_F_part_nufft(u, v, vis, weight, xnew.reshape(imshape))
            Fval = np.square(np.abs(yAx)).sum() / 2

            if lambda_tsv > 0.0:
                # add TSV term to the cost
                tsvcost = TSV(xnew.reshape(imshape))
                Fval += lambda_tsv * tsvcost

            Qval = calc_Q_part(xnew, zvec, c, dfdx.ravel())
            print(f"Qcore: {Qcore}, Qval: {Qval}")
            Qval += Qcore

            print(f"Fval {Fval} Qval {Qval}")
            if Fval <= Qval:
                break

            c *= eta

        print(f"iter {iter_cnt}, xnew mean {xnew.mean()}, std {xnew.std()}")

        eta = ETA  # keep same ETA constant name as in C++ context
        c /= eta

        munew = (1.0 + np.sqrt(1.0 + 4.0 * mu * mu)) / 2.0

        l1cost = np.sum(np.abs(xnew))
        Fval += lambda_l1 * l1cost

        # matches C++ `zvec = xvec;` here: the momentum update below must
        # combine xnew with the *previous xvec*, not the previous zvec
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
                print("stop iteration because xvec is all zero")
                break

        # stopping condition from C++ (uses MINITER and TD)
        if (iter_cnt >= MINITER) \
           and (cost[iter_cnt - TD] - cost[iter_cnt] < eps):
            print(f"converged at iter {iter_cnt}")
            break

        mu = munew

    # end main loop

    # adjust final iter value same as C++ behavior
    if iter_cnt + 1 == maxiter:
        print(f"{iter_cnt:5d} cost = {cost[iter_cnt-1]:.5f}")
        iter_out = iter_cnt
    else:
        print(f"{iter_cnt+1:5d} cost = {cost[iter_cnt]:.5f}")
        iter_out = iter_cnt

    print()

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
        xvec=xvec.reshape(imshape)
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
            maxiter: int = 50000, eps: float = 1.0e-5, cl_box: np.ndarray | None = None):
        """
        Run MFISTA routine to get an image
        """
        # input summary
        print('lambda_l1 = {0}'.format(self.lambda_L1))
        print('lambda_tv = {0}'.format(self.lambda_TV))
        print('lambda_tsv = {0}'.format(self.lambda_TSV))
        print('c = {0:g}'.format(self.cinit))
        print('')
        print('number of u-v points: {0}'.format(inputs.m))
        print('X-dim of image:       {0}'.format(inputs.nx))
        print('Y-dim of image:       {0}'.format(inputs.ny))

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
            xvec=result.xinit.reshape((inputs.nx, inputs.ny))
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
        print('')
        print('')
        print('Input/Output summary:')
        print('')
        print(f' Input u-v data file:      {inputs.infile}')
        if initialimage is None:
            print(' x was initialized with 0.0')
        else:
            print(' x was initialize by the user')
        print('')

    def _show_result(self, mfista_result):
        # show results
        print('')
        print('')
        # print('Output of {0}.'.format(self.libname))
        # print('')
        # print('')
        print(' Size of the problem:')
        print('')
        print('')
        print(' size of input vector:  {0}'.format(mfista_result.M))
        print(' size of output vector: {0}'.format(mfista_result.N))
        if mfista_result.NX != 0:
            print('size of image:          {0} x {1}'.format(mfista_result.NX,
                                                             mfista_result.NY))
        print('')
        print('')
        print(' Problem Setting:')
        print('')
        print('')
        if mfista_result.nonneg == 1:
            print(' x is a nonnegative vector.')
        elif mfista_result.nonneg == 0:
            print(' x is a real vector (takes 0, positive, and negative value).')
        print('')
        print('')
        if mfista_result.lambda_l1 != 0:
            print(' Lambda_l1: {0:e}'.format(mfista_result.lambda_l1))
        if mfista_result.lambda_tsv != 0:
            print(' Lambda_tsv: {0:e}'.format(mfista_result.lambda_tsv))
        if mfista_result.lambda_tv != 0:
            print(' Lambda_tv: {0:e}'.format(mfista_result.lambda_tv))
        print(' MAXITER: {0}'.format(mfista_result.maxiter))

        print(' Results:')
        print('')
        print(' # of iterations:       {0}'.format(mfista_result.ITER))
        print(' cost:                  {0:e}'.format(mfista_result.finalcost))
        print(' computation time[sec]: {0:e}'.format(mfista_result.comp_time))
        print('')
        print(' # of nonzero pixels:   {0}'.format(mfista_result.N_active))
        print(' Squared Error (SE):    {0:e}'.format(mfista_result.sq_error))
        print(' Mean SE:               {0:e}'.format(mfista_result.mean_sq_error))
        if mfista_result.lambda_l1 != 0:
            print(' L1 cost:               {0:e}'.format(mfista_result.l1cost))
        if mfista_result.lambda_tsv != 0:
            print(' TSV cost:              {0:e}'.format(mfista_result.tsvcost))
        if mfista_result.lambda_tv != 0:
            print(' TV cost:               {0:e}'.format(mfista_result.tvcost))
        print('')
        print(' LOOE:    Could not be computed because Hessian was not positive definite.')
