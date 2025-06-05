import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1, rect):
    """
    Inverse compositional affine Lucas-Kanade (Q3.3)
    Inputs:
        It   : Template frame (grayscale)
        It1  : Current frame (grayscale)
        rect : [x1, y1, x2, y2] region in It
    Output:
        M    : 2x3 affine warp matrix mapping It to It1
    """
    convergence_eps = 0.01875
    max_iters = 100
    dtype = np.float64

    # Initialize warp matrix as identity
    warp_matrix = np.eye(3, dtype=dtype)
    x1, y1, x2, y2 = map(float, rect)

    h, w = It.shape
    x_vals = np.arange(w)
    y_vals = np.arange(h)
    interp_template = RectBivariateSpline(x_vals, y_vals, It.T)
    interp_frame    = RectBivariateSpline(x_vals, y_vals, It1.T)

    nX = max(1, int(round(x2 - x1)))
    nY = max(1, int(round(y2 - y1)))
    grid_x = np.linspace(x1, x2, nX)
    grid_y = np.linspace(y1, y2, nY)
    GX, GY = np.meshgrid(grid_x, grid_y)

    template_vals = interp_template.ev(GX, GY).ravel()
    dTx = interp_template.ev(GX, GY, dx=1, dy=0).ravel()
    dTy = interp_template.ev(GX, GY, dx=0, dy=1).ravel()

    Xf, Yf = GX.ravel(), GY.ravel()
    jacobian = np.vstack([
        dTx * Xf,
        dTx * Yf,
        dTx,
        dTy * Xf,
        dTy * Yf,
        dTy
    ]).T

    H = jacobian.T @ jacobian
    H += np.eye(6, dtype=dtype) * 1e-6  # Regularization
    H_inv = np.linalg.pinv(H)

    ones = np.ones_like(Xf)
    coords = np.vstack([Xf, Yf, ones])

    for _ in range(max_iters):
        warped_coords = warp_matrix @ coords
        Xw = warped_coords[0].reshape(GX.shape)
        Yw = warped_coords[1].reshape(GY.shape)

        warped_vals = interp_frame.ev(Xw, Yw).ravel()
        error = warped_vals - template_vals

        deltaP = H_inv @ jacobian.T @ error
        if np.linalg.norm(deltaP) < convergence_eps:
            break

        update = np.array([
            [1 + deltaP[0], deltaP[1],   deltaP[2]],
            [deltaP[3],     1 + deltaP[4], deltaP[5]],
            [0, 0, 1]
        ], dtype=dtype)

        warp_matrix = warp_matrix @ np.linalg.inv(update)

    return warp_matrix[:2, :]
