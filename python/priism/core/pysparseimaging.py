from __future__ import annotations

from functools import partial
from collections.abc import Callable

import finufft
import numpy as np


# MSP = 6
# MSP2 = MSP * 2
# MSP4 = MSP * 4
MAXITER = 50000
MINITER = 100
TD = 50
ETA = 1.1


def soft_thresold(v: float, eta: float) -> float:
    """Apply soft-thresholding to a value.

    Args:
        v: input value
        eta: threshold value
    Returns:
        Soft-thresholded value
    """
    abs_eta = abs(eta)

    r = 0.0
    if v > abs_eta:
        r = v - abs_eta
    elif v < -abs_eta:
        r = v + abs_eta

    return r


def soft_thresold_nonneg(v: float, eta: float) -> float:
    """Apply soft-thresholding to a non-negative value.

    Args:
        v: input value
        eta: threshold value
    Returns:
        Soft-thresholded value
    """
    abs_eta = abs(eta)

    r = 0.0
    if v > abs_eta:
        r = v - abs_eta

    return r


def soft_threshold_box(vec: np.ndarray, eta: float, box_flag: bool, box: np.ndarray | None, threshold_func: Callable[[float, float], float]) -> np.ndarray:
    """Apply soft-thresholding with box constraints.

    Args:
        vec: input numpy array
        eta: threshold value
        box_flag: if True, apply box constraints
        box: box constraints array (same shape as vec) or None
        threshold_func: function to apply for thresholding

    Returns:
        Soft-thresholded numpy array
    """
    nvec = np.asarray([threshold_func(v, eta) for v in vec], dtype=float)

    if box_flag:
        nvec *= box

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
    term1 = np.matmul(buf_vec, AyAz)
    # (x1 - x2)'(x1 - x2)
    term2 = np.square(buf_vec)

    return -term1 + c * term2 / 2


def calc_F_part_nufft(u: np.ndarray, v: np.ndarray, vis: np.ndarray, weight: np.ndarray, xvec: np.ndarray) -> np.ndarray:
    """Calculate the F part of the cost function using NUFFT.

    Args:
        u: u coordinates of visibilities (length M)
        v: v coordinates of visibilities (length M)
        vis: observed visibilities (length M)
        weight: visibility weights (length M)
        xvec: model image array (length Nx*Ny)

    Returns:
        yAx: visibility difference array (length M)
    """
    model_vis = finufft.nufft2d2(u, v, xvec)
    yAx = (model_vis - vis) * weight

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
    weighted_yAx = yAx * weight
    dfdx = finufft.nufft2d1(u, v, weighted_yAx, n_modes=(nx, ny))

    return dfdx


def mfista_L1_TSV_core_nufft(
        xout: np.ndarray,
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
) -> tuple[int, float]:
    """Python translation of the C++ mfista_L1_TSV_core_nufft.

    Args:
        xout: numpy array (length Nx*Ny) to be filled inplace
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
    """

    NN = Nx * Ny

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
        return 0, 0.0

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
    yAx = calc_F_part_nufft(u, v, vis, weight, xvec)
    costtmp = np.square(yAx).sum() / 2

    # looks like equivalent is
    #   1. perform NUFFT to get model visibilities from xvec
    #   2. compute (model - observed) * weight
    #   3. compute 0.5 * norm squared of the result

    # add L1 normalizaton term to the cost
    l1cost = np.sum(np.abs(xvec))
    costtmp += lambda_l1 * l1cost

    # add TSV term to the cost
    if lambda_tsv > 0:
        tsvcost = TSV(xvec)
        costtmp += lambda_tsv * tsvcost

    eta = 10.0
    iter_cnt = 0
    mu = 1.0

    for iter_cnt in range(maxiter):
        cost[iter_cnt] = costtmp

        if (iter_cnt % 10) == 0:
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
        yAx = calc_F_part_nufft(u, v, vis, weight, zvec)
        Qcore = np.square(yAx).sum() / 2

        # compute gradient dF/dx at zvec
        # input:
        #   yAx: visibility diff, (model - observed) * weight
        # output:
        #   dfdx: Fourier transform of visibility gradient, (model - observed) * weight**2
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
            tsvcost = TSV(zvec)
            Qcore += lambda_tsv * tsvcost

            # compute gradient of TSV term
            d_TSV(zvec)
            dfdx = dfdx - lambda_tsv * dtmp

        # inner loop for line-search / backtracking
        Fval = 0.0
        for _ in range(maxiter):
            xtmp = zvec + dfdx / c
            # no box_flag
            #   xnew = max(xtmp - lambda_l1 / c, 0)
            # with box_flag
            #   xnew = max(xtmp - lambda_l1 / c, 0) if box > 0 else 0
            soft_th_box(xnew, xtmp, lambda_l1 / c, box_flag, box)

            # compute cost at xnew
            # Fval = calc_F_part_nufft(
            #     M, Nx, Ny, yAx, E1, E2x, E2y, E4, mx, my, y_neg,
            #     rvec, cvec, fftwplan_r2c, vis, weight, xnew, mbuf_h
            # )
            yAx = calc_F_part_nufft(u, v, vis, weight, xnew)
            Fval = np.square(yAx).sum() / 2

            if lambda_tsv > 0.0:
                # add TSV term to the cost
                tsvcost = TSV(xnew)
                Fval += lambda_tsv * tsvcost

            Qval = calc_Q_part(xnew, zvec, c, dfdx, xtmp)
            Qval += Qcore

            if Fval <= Qval:
                break

            c *= eta

        eta = ETA  # keep same ETA constant name as in C++ context
        c /= eta

        munew = (1.0 + np.sqrt(1.0 + 4.0 * mu * mu)) / 2.0

        l1cost = np.sum(np.abs(xnew))
        Fval += lambda_l1 * l1cost

        z_old = zvec.copy()

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
                break

        # stopping condition from C++ (uses MINITER and TD)
        converged = (cost[iter_cnt - TD] - cost[iter_cnt]) < eps
        if (iter_cnt >= MINITER) and converged:
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

    # copy result to xout inplace
    xout[:] = xvec[:]

    return iter_out + 1, c
