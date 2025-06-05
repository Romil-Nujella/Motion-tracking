import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter

def compute_weights(error, alpha=None, method="huber", return_diag=False):
    if method == "huber":
        alpha = alpha or 1.339
        weights = np.tile(alpha, error.shape[0]) / np.abs(error)
    elif method == "tukey":
        alpha = alpha or 4.685
        weights = (1 - (error / alpha) ** 2) ** 3
    else:
        weights = np.ones_like(error)

    np.clip(weights, 0, 1, out=weights)
    return np.diag(weights) if return_diag else weights

def LucasKanadeRobust(It, It1, rect, num_levels=3, scale=0.5,
                      lk_thresh=0.03, lk_maxiter=50, method="huber"):
    # Build image pyramids from coarsest to finest
    pyramid_It, pyramid_It1 = [It], [It1]
    for _ in range(1, num_levels):
        pyramid_It.insert(0, gaussian_filter(pyramid_It[0], 1)[::2, ::2])
        pyramid_It1.insert(0, gaussian_filter(pyramid_It1[0], 1)[::2, ::2])

    p = np.zeros(2)
    scaled_rect = np.array(rect) * (scale ** (num_levels - 1))

    for lvl in range(num_levels):
        img_t = pyramid_It[lvl]
        img_t1 = pyramid_It1[lvl]
        h, w = img_t.shape

        x1, y1, x2, y2 = scaled_rect + np.tile(p, 2)
        x1, x2 = np.clip([x1, x2], 0, w - 1)
        y1, y2 = np.clip([y1, y2], 0, h - 1)

        interp_t = RectBivariateSpline(np.arange(w), np.arange(h), img_t.T)
        interp_i = RectBivariateSpline(np.arange(w), np.arange(h), img_t1.T)

        nx, ny = int(round(x2 - x1)), int(round(y2 - y1))
        gx = np.linspace(x1, x2, nx)
        gy = np.linspace(y1, y2, ny)
        GX, GY = np.meshgrid(gx, gy)
        T_vals = interp_t.ev(GX, GY).ravel()

        coords_x, coords_y = GX.ravel(), GY.ravel()

        for _ in range(lk_maxiter):
            X_new = coords_x + p[0]
            Y_new = coords_y + p[1]
            I_vals = interp_i.ev(X_new, Y_new).ravel()
            err = T_vals - I_vals

            Ix = interp_i.ev(X_new, Y_new, dx=1).ravel()
            Iy = interp_i.ev(X_new, Y_new, dy=1).ravel()

            A = np.vstack([Ix, Iy]).T
            weights = compute_weights(err, method=method)
            sqrt_w = np.sqrt(weights)

            A_weighted = A * sqrt_w[:, None]
            b_weighted = err * sqrt_w

            H = A_weighted.T @ A_weighted
            delta_p = np.linalg.lstsq(H, A_weighted.T @ b_weighted, rcond=None)[0]

            p += delta_p
            if np.linalg.norm(delta_p) < lk_thresh:
                break

        if lvl < num_levels - 1:
            p /= scale
            scaled_rect /= scale

    return p[0], p[1]
