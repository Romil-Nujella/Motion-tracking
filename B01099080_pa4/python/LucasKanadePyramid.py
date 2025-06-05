import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectBivariateSpline

def LucasKanadePyramid(It, It1, rect, num_levels=3, scale=0.5,
                       method="huber", normalize=True,
                       lk_thresh=0.02, lk_maxiter=50):

    # Normalize brightness if required
    if normalize:
        x1, y1, x2, y2 = map(int, rect)
        template_mean = np.mean(It[y1:y2+1, x1:x2+1])
        current_mean = np.mean(It1[y1:y2+1, x1:x2+1])
        if current_mean > 0:
            It1 = It1.astype(np.float64) * (template_mean / current_mean)


    # Build multi-level pyramids
    pyr_template, pyr_image = [], []
    temp, img = It.copy(), It1.copy()
    for _ in range(num_levels):
        pyr_template.insert(0, temp)
        pyr_image.insert(0, img)
        temp = gaussian_filter(temp, 1)[::2, ::2]
        img = gaussian_filter(img, 1)[::2, ::2]

    p = np.zeros(2)
    scaled_rect = np.array(rect) * (scale ** (num_levels - 1))

    for lvl in range(num_levels):
        tmpl = pyr_template[lvl]
        curr = pyr_image[lvl]
        h, w = tmpl.shape

        rx1, ry1, rx2, ry2 = scaled_rect + np.tile(p, 2)

        xs = np.arange(w)
        ys = np.arange(h)
        interp_tmpl = RectBivariateSpline(xs, ys, tmpl.T)
        interp_curr = RectBivariateSpline(xs, ys, curr.T)

        nx = max(int(round(rx2 - rx1)), 1)
        ny = max(int(round(ry2 - ry1)), 1)
        gx = np.linspace(rx1, rx2, nx)
        gy = np.linspace(ry1, ry2, ny)
        GX, GY = np.meshgrid(gx, gy)
        T_vals = interp_tmpl.ev(GX, GY).ravel()

        for _ in range(lk_maxiter):
            X_shift = GX + p[0]
            Y_shift = GY + p[1]
            I_vals = interp_curr.ev(X_shift, Y_shift).ravel()
            error = T_vals - I_vals

            Ix = interp_curr.ev(X_shift, Y_shift, dx=1).ravel()
            Iy = interp_curr.ev(X_shift, Y_shift, dy=1).ravel()

            SD_images = np.vstack([Ix, Iy]).T

            if method == "huber":
                alpha = 1.339
                weights = np.clip(alpha / np.abs(error), 0, 1)
            elif method == "tukey":
                alpha = 4.685
                weights = np.clip((1 - (error / alpha) ** 2) ** 3, 0, 1)
            else:
                weights = np.ones_like(error)

            sqrt_weights = np.sqrt(weights)[:, None]
            A_weighted = SD_images * sqrt_weights
            b_weighted = error * sqrt_weights.ravel()
            H = A_weighted.T @ A_weighted
            delta_p = np.linalg.lstsq(H, A_weighted.T @ b_weighted, rcond=None)[0]

            p += delta_p
            if np.linalg.norm(delta_p) < lk_thresh:
                break

        if lvl < num_levels - 1:
            p /= scale
            scaled_rect /= scale

    return p[0], p[1]
