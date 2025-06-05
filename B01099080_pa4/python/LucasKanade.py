import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect):
    """
    Q3.1
        [I] It: Template image
            It1: Current image
            rect: Current position of the object
                (top left, bottom right coordinates: x1, y1, x2, y2)
        [O] p: movement vector dx, dy
    """

    # Set up the threshold
    threshold = 0.01875
    maxIters = 100
    npDtype = np.float64    # Might be useful
    # p := dx, dy
    p = np.zeros(2, dtype=npDtype)  # OR p = np.zeros(2)
    x1, y1, x2, y2 = rect

    # Crop template image
    height, width = It.shape
    _x, _y = np.arange(width), np.arange(height)

    # This returns a class object; note the swap/transpose
    # Use spline.ev() for getting values at locations
    splineT = RectBivariateSpline(_x, _y, It.T)
    splineI = RectBivariateSpline(_x, _y, It1.T)

    nX, nY = int(x2 - x1), int(y2 - y1)
    coordsX = np.linspace(x1, x2, nX, dtype=npDtype)
    coordsY = np.linspace(y1, y2, nY, dtype=npDtype)

    # Generate coordinate grid for the template
    X, Y = np.meshgrid(coordsX, coordsY)
    template_patch = splineT.ev(X, Y)

    for _ in range(maxIters):
        # 1. Warp coordinates
        X_warped = X + p[0]
        Y_warped = Y + p[1]

        # 2. Evaluate warped image at those coordinates
        current_patch = splineI.ev(X_warped, Y_warped)

        # Compute error between template and warped patch
        diff = template_patch - current_patch

        # Compute gradients of warped image
        grad_x = splineI.ev(X_warped, Y_warped, dx=1, dy=0)
        grad_y = splineI.ev(X_warped, Y_warped, dx=0, dy=1)

        # Flatten gradients and error
        gx = grad_x.ravel()
        gy = grad_y.ravel()
        diff_flat = diff.ravel()

        # Compute Hessian matrix
        H = np.array([
            [np.sum(gx * gx), np.sum(gx * gy)],
            [np.sum(gx * gy), np.sum(gy * gy)]
        ])

        # Compute b vector
        b = np.array([np.sum(gx * diff_flat), np.sum(gy * diff_flat)])

        # Solve for parameter update
        deltaP = np.linalg.lstsq(H, b, rcond=None)[0]
        p += deltaP

        if np.linalg.norm(deltaP) < threshold:
            break

    return p
