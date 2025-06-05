import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1, rect):
    """
    Lucas-Kanade with affine transformation (Q3.2)
    Inputs:
        It   : Grayscale template image
        It1  : Grayscale next frame
        rect : Bounding box [x1, y1, x2, y2] in template
    Output:
        M    : 2x3 affine matrix transforming It to It1
    """
    max_iterations = 100
    convergence_eps = 0.01875
    p = np.zeros((6, 1), dtype=np.float64)

    x1, y1, x2, y2 = rect
    h, w = It.shape
    x = np.arange(w)
    y = np.arange(h)
    spline_template = RectBivariateSpline(x, y, It.T)
    spline_current  = RectBivariateSpline(x, y, It1.T)

    nx = max(1, int(round(x2 - x1)))
    ny = max(1, int(round(y2 - y1)))
    gx = np.linspace(x1, x2, nx)
    gy = np.linspace(y1, y2, ny)
    GX, GY = np.meshgrid(gx, gy)
    template_vals = spline_template.ev(GX, GY).ravel()
    Xf, Yf = GX.ravel(), GY.ravel()
    ones = np.ones_like(Xf)

    for _ in range(max_iterations):
        affine_matrix = np.array([
            [1 + p[0][0], p[1][0], p[2][0]],
            [p[3][0], 1 + p[4][0], p[5][0]]
        ])
        coords = np.vstack([Xf, Yf, ones])
        warped_coords = affine_matrix @ coords
        Xw = warped_coords[0].reshape(GX.shape)
        Yw = warped_coords[1].reshape(GY.shape)

        Iw = spline_current.ev(Xw, Yw).ravel()
        error = template_vals - Iw

        Ix = spline_current.ev(Xw, Yw, dx=1, dy=0).ravel()
        Iy = spline_current.ev(Xw, Yw, dx=0, dy=1).ravel()

        SD = np.vstack([
            Ix * Xf,
            Ix * Yf,
            Ix,
            Iy * Xf,
            Iy * Yf,
            Iy
        ]).T

        H = SD.T @ SD
        dp = np.linalg.lstsq(H, SD.T @ error, rcond=None)[0].reshape(-1, 1)
        p += dp

        if np.linalg.norm(dp) < convergence_eps:
            break

    final_matrix = np.array([
        [1 + p[0][0], p[1][0], p[2][0]],
        [p[3][0], 1 + p[4][0], p[5][0]]
    ])
    return final_matrix
