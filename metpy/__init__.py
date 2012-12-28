import version
import kde
import numpy as np


__all__ = ['kde']

__version__ = version.get_version()


def smth9(f, p=0.5, q=0):
    """
    Performs nine point local smoothing on a 2D grid. This is a
    Python port of the routine described here:

        http://www.ncl.ucar.edu/Document/Functions/Built-in/smth9.shtml

    In general, for good results, set p = 0.50. With p = 0.50, a value of
    q = -0.25 results in "light" smoothing and q = 0.25 results in "heavy"
    smoothing. A value of q = 0.0 results in a 5-point local smoother.

    Parameters
    ----------
    f : array_like
        A grid of two dimensions.
    p : float optional (default: 0.5)
    q : float optional (default: 0)

    Returns
    -------
    f : array_like
        The smoothed array_like

    """
    f = np.asarray(f)
    f[1:-1, 1:-1] += (p / 4) * (f[0:-2, 1:-1] + f[1:-1, 0:-2] +
                     f[2:, 1:-1] + f[1:-1, 2:] - 4 * f[1:-1, 1:-1])
    f[1:-1, 1:-1] += (q / 4) * (f[0:-2, 0:-2] + f[0:-2, 2:] +
                     f[2:, 2:] + f[2:, 0:-2] - 4 * f[1:-1, 1:-1])
    return f