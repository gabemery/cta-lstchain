import numpy as np
from ctapipe.image.toymodel import Gaussian, SkewedGaussian
import timeit

def log_gaussian(x, mean, sigma):
    """
    Evaluate the log of a normal law

    Parameters
    ----------
    x: float or array-like
        Value at which the log gaussian is evaluated 
    mean: float
        Central value of the normal distribution
    sigma: float
        Width of the normal distribution

    Returns
    -------
    log_pdf: float or array-like
        Log of the evaluation of the normal law at x
    """

    log_pdf = -(x - mean) ** 2 / (2 * sigma ** 2)
    log_pdf = log_pdf - np.log((np.sqrt(2 * np.pi) * sigma))

    return log_pdf


def log_gaussian2d(size, x, y, x_cm, y_cm, width, length, psi):
    """
    Evaluate the log of a bi-dimensionnal gaussian law

    Parameters
    ----------
    size: float
        Integral of the 2D Gaussian
    x, y: float or array-like
        Position at which the log gaussian is evaluated 
    x_cm, y_cm: float
        Center of the 2D Gaussian
    width, length: float
        Standard deviations of the 2 dimensions of the 2D Gaussian law
    psi: float
        Orientation of the 2D Gaussian

    Returns
    -------
    log_pdf: float or array-like
        Log of the evaluation of the 2D gaussian law at (x,y)

    """
    scale_w = 1. / (2. * width ** 2)
    scale_l = 1. / (2. * length ** 2)
    a = np.cos(psi) ** 2 * scale_l + np.sin(psi) ** 2 * scale_w
    b = np.sin(2 * psi) * (scale_w - scale_l) / 2.
    c = np.cos(psi) ** 2 * scale_w + np.sin(psi) ** 2 * scale_l

    norm = 1. / (2 * np.pi * width * length)

    log_pdf = - (a * (x - x_cm) ** 2 - 2 * b * (x - x_cm) * (y - y_cm) + c * (
                y - y_cm) ** 2)

    log_pdf += np.log(norm) + np.log(size)

    return log_pdf


def log_gaussian2d_ctapipe(size, x, y, x_cm, y_cm, width, length, psi):
    gaussian = Gaussian(x_cm, y_cm, length, width, psi)
    log_pdf = np.log(gaussian.pdf(x, y)) + np.log(size)
    return log_pdf


def log_gaussian2d_skewed(size, x, y, x_cm, y_cm, width, length, psi, skewness):
    skewed_gaussian = SkewedGaussian(x_cm, y_cm, length, width, psi, skewness)
    log_pdf = np.log(skewed_gaussian.pdf(x, y)) + np.log(size)
    return log_pdf

if __name__ == '__main__':
    print(timeit.timeit("log_gaussian2d(10,"
                        " np.asarray([0,0.1,0.2]),"
                        " np.asarray([0,0.1,0.2]),"
                        "0.1,0.1,0.1,0.3,1)", setup="from __main__ import log_gaussian2d\nimport numpy as np",
                        number=1000000
                        ))

    print(timeit.timeit("log_gaussian2d_ctapipe(10,"
                        " np.asarray([0,0.1,0.2])*u.m,"
                        " np.asarray([0,0.1,0.2])*u.m,"
                        "0.1*u.m,0.1*u.m,0.1*u.m,0.3*u.m,1*u.rad)", setup="from __main__ import log_gaussian2d_ctapipe\nimport numpy as np\nimport astropy.units as u",
                        number=1000000
                        ))
