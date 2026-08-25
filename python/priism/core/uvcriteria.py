# Copyright (C) 2026
# The Institute of Statistical Mathematics
# 10-3 Midori-cho, Tachikawa, Tokyo 190-8562, Japan.
#
# This file is part of PRIISM.
#
# PRIISM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# PRIISM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with PRIISM.  If not, see <https://www.gnu.org/licenses/>.
"""
u-v-distance-based evaluation criteria for choosing the L1/TSV weights,
as an alternative to cross-validation (Ikeda et al. 2025, PASJ 77(2):260-276,
section 3.4, eq. 15-17).

Ported from old/python/sparseimaging/assess_results.py (min_ellipsoid*,
ellipsoid_map, inside_power_min_ellipsoid, errors_cos, hinge,
comp_errors_mfista), adapted from the old radian-scaled ImagingUVData
convention to priism's pixel-grid u/v convention (VisibilityWorkingSet.u/v
are grid-pixel coordinates with the origin at nx/2, ny/2 -- see
alma/visconverter.py::fill_uvw).

C1 (u-v ellipsoid power ratio) and C2 (grouped-residual cosine similarity)
are treated as soft constraints (via a squared-hinge penalty, zero once
both are above their thresholds) rather than being directly minimized;
among (L1, Ltsv) satisfying both, the weighted mean-squared visibility
residual is minimized. This matches how old/assess_results.py's
comp_errors_mfista combines the two criteria, rather than a naive
additive (1-C1)+(1-C2) blend.

Unlike cross-validation, both C1 and C2 are computed directly from a
single full-data MFISTA solve's residual, so no repeated CV re-estimation
is needed -- this is what makes the criterion cheap enough to combine
efficiently with Bayesian Optimization.
"""
from __future__ import absolute_import

import numpy as np
import scipy.interpolate


def _initial_points(u, v):
    s = u ** 2
    t = v ** 2
    dist2 = s + t
    idx1 = int(np.argmax(dist2))
    u1, v1 = u[idx1], v[idx1]
    a1 = np.sqrt(dist2[idx1])

    cos1 = u1 / a1
    sin1 = v1 / a1

    b1 = 0.0
    idx2 = None
    mask = np.ones(u.size, dtype=bool)
    mask[idx1] = False
    A = 1.0 - ((u[mask] * cos1 + v[mask] * sin1) / a1) ** 2
    valid = A > 0
    if np.any(valid):
        B = (u[mask][valid] * sin1 - v[mask][valid] * cos1) ** 2
        b = np.sqrt(B / A[valid])
        local_idx = int(np.argmax(b))
        if b[local_idx] > b1:
            b1 = b[local_idx]
            idx2 = np.nonzero(mask)[0][valid][local_idx]

    return idx1, idx2


def _min_ellipsoid_2pnts(u1, v1, u2, v2):
    """A*u^2 + B*v^2 + 2*C*u*v = 1 through two points (minimal-area choice)."""
    s1, s2 = u1 * u1, u2 * u2
    t1, t2 = v1 * v1, v2 * v2
    w1, w2 = u1 * v1, u2 * v2

    if u1 * v2 == u2 * v1:
        return 0.0, 0.0, 0.0
    elif (u1 * u2 + v1 * v2) == 0.0 and t1 + s1 == t2 + s2:
        A = 1.0 / (s1 + t1)
        return A, A, 0.0
    elif (t1 * s2 - t2 * s1) == 0.0:
        A = (t1 + t2) / (4 * s1 * t2)
        B = (t1 + t2) / (4 * t1 * t2)
        C = (v1 / (4 * u1)) * ((t2 - t1) / (t1 * t2))
        return A, B, C
    else:
        alpha1 = t1 - t2
        alpha2 = s2 - s1
        beta1 = w1 * t2 - w2 * t1
        beta2 = w2 * s1 - w1 * s2

        denom0 = (t1 * s2 - t2 * s1) ** 2 - 4 * beta1 * beta2
        C = (alpha1 * beta2 + alpha2 * beta1) / denom0

        A = (alpha1 + 2 * beta1 * C) / (t1 * s2 - t2 * s1)
        B = (alpha2 + 2 * beta2 * C) / (t1 * s2 - t2 * s1)
        return A, B, C


def _min_ellipsoid_3pnts(u1, v1, u2, v2, u3, v3):
    mat = np.array([
        [u1 * u1, v1 * v1, 2 * u1 * v1],
        [u2 * u2, v2 * v2, 2 * u2 * v2],
        [u3 * u3, v3 * v3, 2 * u3 * v3],
    ])
    if abs(np.linalg.det(mat)) <= 1.0e-8:
        return 0.0, 0.0, 0.0
    A, B, C = np.linalg.solve(mat, np.ones(3))
    if A <= 0 or B <= 0 or A * B - C * C <= 0:
        return 0.0, 0.0, 0.0
    return A, B, C


def _ellipsoid_area(A, B, C):
    if A > 0 and B > 0 and (A * B - C * C) > 0:
        return np.pi / np.sqrt(A * B - C * C)
    return 0.0


def _min_ellipsoid_sub(u, v, s, t, w, active):
    idx = np.nonzero(active)[0]
    n = idx.size
    uu, vv = u[idx], v[idx]

    candidates = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            a, b, c = _min_ellipsoid_2pnts(uu[i], vv[i], uu[j], vv[j])
            if a > 0 and b > 0:
                contour = a * s + b * t + 2 * c * w - 1
                contour[idx[i]] = -1
                contour[idx[j]] = -1
                if np.all(contour <= 0):
                    candidates.append((a, b, c, [idx[i], idx[j]]))

    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                a, b, c = _min_ellipsoid_3pnts(
                    uu[i], vv[i], uu[j], vv[j], uu[k], vv[k]
                )
                if a > 0 and b > 0:
                    contour = a * s + b * t + 2 * c * w - 1
                    contour[idx[i]] = -1
                    contour[idx[j]] = -1
                    contour[idx[k]] = -1
                    if np.all(contour <= 0):
                        candidates.append((a, b, c, [idx[i], idx[j], idx[k]]))

    best = min(candidates, key=lambda cand: _ellipsoid_area(cand[0], cand[1], cand[2]))
    return best


def min_ellipsoid(u, v):
    """Minimum covering ellipsoid A*u^2 + B*v^2 + 2*C*u*v = 1 of the given
    u-v points. Mirrors old/python/sparseimaging/assess_results.py::min_ellipsoid,
    generalized to take plain (u, v) arrays in whatever linear units the
    caller uses (pixel-grid units for priism's VisibilityWorkingSet).
    """
    max_active = 100

    s = u ** 2
    t = v ** 2
    w = u * v
    n = u.size

    idx1, idx2 = _initial_points(u, v)
    supporting = [idx1, idx2]
    A, B, C = _min_ellipsoid_2pnts(u[idx1], v[idx1], u[idx2], v[idx2])

    critical = np.zeros(n, dtype=bool)

    while True:
        contour = A * s + B * t + 2 * C * w - 1
        for i in supporting:
            contour[i] = -1

        if np.all(contour <= 0):
            break

        for i in supporting:
            contour[i] = 0
            critical[i] = True

        sort_idx = np.argsort(-contour)
        active = np.zeros(n, dtype=bool)
        top = sort_idx[:max_active]
        active[top[contour[top] > 0]] = True
        active[critical] = True

        A, B, C, supporting = _min_ellipsoid_sub(u, v, s, t, w, active)

    return A, B, C


def ellipsoid_map(A, B, C, nx, ny):
    """Boolean (nx, ny)-shaped mask: True where the FFT frequency-grid
    pixel (centered at nx/2, ny/2, matching VisibilityWorkingSet's u/v
    convention) lies inside the ellipsoid A*u^2 + B*v^2 + 2*C*u*v = 1.
    """
    x_idx = np.arange(nx, dtype=np.float64) - nx / 2.0
    y_idx = np.arange(ny, dtype=np.float64) - ny / 2.0
    xx, yy = np.meshgrid(x_idx, y_idx, indexing='ij')
    return (A * xx ** 2 + B * yy ** 2 + 2 * C * xx * yy) <= 1.0


def _hinge(x):
    """max(x, 0), mirrors old/python/sparseimaging/assess_results.py::hinge."""
    return x if x > 0 else 0.0


def inside_power_ratio(image_2d, mask):
    """Fraction of image power (in the Fourier domain) that falls inside
    the covering-ellipsoid mask -- this is C1.
    """
    imagefft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(image_2d)))
    power = np.abs(imagefft) ** 2
    total = power.sum()
    if total <= 0:
        return 0.0
    return float((power * mask).sum() / total)


def _model_visibility_from_image(image_2d, u, v):
    """Model visibility at the given (u, v) grid-pixel coordinates,
    obtained by FFT of the image and bicubic interpolation -- same
    technique as cv.MeanSquareErrorEvaluator._evaluate_mse.
    """
    shifted_imagefft = np.fft.fft2(np.fft.fftshift(image_2d))
    imagefft = np.fft.ifftshift(shifted_imagefft).transpose()
    nx, ny = imagefft.shape
    rinterp = scipy.interpolate.RectBivariateSpline(np.arange(nx), np.arange(ny), imagefft.real)
    iinterp = scipy.interpolate.RectBivariateSpline(np.arange(nx), np.arange(ny), imagefft.imag)
    rmodel = rinterp(v, u, grid=False)
    imodel = iinterp(v, u, grid=False)
    return rmodel, imodel


def grouped_residual_cosine(u, v, rdata, idata, weight, image_2d):
    """C2: split visibilities into 3 groups by u-v distance from the
    origin, compute the weighted-mean-squared residual error in each
    group, and take the cosine similarity of the resulting 3-vector
    against (1, 1, 1) -- maximal (1.0) when the error is evenly spread
    across baseline lengths, which is the criterion's balance condition
    between the L1 (favors low-frequency error) and TSV (favors
    high-frequency error) regularizers.

    Mirrors old/python/sparseimaging/assess_results.py::errors_cos.
    """
    rmodel, imodel = _model_visibility_from_image(image_2d, u, v)

    error = weight * ((rdata - rmodel) ** 2 + (idata - imodel) ** 2) / 2.0
    uvdist = np.sqrt(u ** 2 + v ** 2)

    max_uv = uvdist.max()
    mean_error = error.mean()

    thres1 = max_uv / 3.0
    thres2 = 2.0 * thres1

    group1 = uvdist < thres1
    group2 = (uvdist >= thres1) & (uvdist < thres2)
    group3 = uvdist >= thres2

    var1 = error[group1].mean() if np.any(group1) else 0.0
    var2 = error[group2].mean() if np.any(group2) else 0.0
    var3 = error[group3].mean() if np.any(group3) else 0.0

    denom = np.sqrt(3.0) * np.sqrt(var1 ** 2 + var2 ** 2 + var3 ** 2)
    cos = (var1 + var2 + var3) / denom if denom > 0 else 0.0

    return cos, mean_error, (var1, var2, var3)


class UvEllipsoidEvaluator(object):
    """Evaluate the C1/C2 criterion for a given working set, computing the
    minimum covering u-v ellipsoid once and reusing it across many trials
    (e.g. within a Bayesian Optimization study).
    """
    def __init__(self, working_set, nx, ny):
        self.nx = nx
        self.ny = ny
        u = np.asarray(working_set.u, dtype=np.float64) - nx / 2.0
        v = np.asarray(working_set.v, dtype=np.float64) - ny / 2.0
        self.u = u
        self.v = v
        A, B, C = min_ellipsoid(u, v)
        self.ellipsoid = (A, B, C)
        self.mask = ellipsoid_map(A, B, C, nx, ny)

    def evaluate(self, working_set, image_2d,
                 ellipse_th=0.99, cos_th=0.99, pow_scale=0.01, cos_scale=0.01):
        """Return (cost, C1, C2) for the given full (non-CV-split) working
        set and resulting image.

        C1 and C2 are treated as soft constraints (C1 >= ellipse_th, C2 >=
        cos_th) via a squared-hinge penalty -- zero once both are satisfied,
        growing quadratically beyond that -- and the remaining cost is the
        weighted mean-squared visibility residual (chisq_mean). So this
        selects, among the (L1, Ltsv) that satisfy the two u-v-distance
        criteria, the one that fits the data best. Mirrors
        old/python/sparseimaging/assess_results.py::comp_errors_mfista.
        """
        u = np.asarray(working_set.u, dtype=np.float64) - self.nx / 2.0
        v = np.asarray(working_set.v, dtype=np.float64) - self.ny / 2.0

        c1 = inside_power_ratio(image_2d, self.mask)
        c2, mean_error, _groups = grouped_residual_cosine(
            u, v, working_set.rdata, working_set.idata, working_set.weight, image_2d
        )

        tmp1 = _hinge((ellipse_th - c1) / pow_scale)
        tmp2 = _hinge((cos_th - c2) / cos_scale)

        cost = 100.0 * (tmp1 ** 2 + tmp2 ** 2) + mean_error

        return cost, c1, c2
