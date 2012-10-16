import numpy as np
from scipy import ndimage

__all__ = ['gauss2d', 'fit_gauss2d']


def fit_gauss2d(data):
    """
    Fit a 2-dimensional anisotropic Gaussian to an image.

    Parameters
    ----------
    data : array_like
        A two dimensional array_like object

    Returns
    -------
    sigx : float
        The standard deviation of the long axis

    sigy : float
        The standard deviation of the short axis

    xrot : float
        The angle between the x-axis and the long axis of the distribution.

    """
    def raw_moment(data, iord, jord):
        nrows, ncols = data.shape
        y, x = np.mgrid[:nrows, :ncols]
        data = data * x**iord * y**jord
        return data.sum()

    def inertial_axis(data):
        """
        Calculate the x-mean, y-mean, and cov matrix of an image.

        """
        data_sum = data.sum()
        m10 = raw_moment(data, 1, 0)
        m01 = raw_moment(data, 0, 1)
        x_bar = m10 / data_sum
        y_bar = m01 / data_sum
        u11 = (raw_moment(data, 1, 1) - x_bar * m01) / data_sum
        u20 = (raw_moment(data, 2, 0) - x_bar * m10) / data_sum
        u02 = (raw_moment(data, 0, 2) - y_bar * m01) / data_sum
        cov = np.array([[u20, u11], [u11, u02]])
        return x_bar, y_bar, cov

    data = np.asarray(data)
    xbar, ybar, cov = inertial_axis(data)
    eigvals, eigvecs = np.linalg.eigh(cov)
    ind_of_max_eigvalue = np.where(eigvals == eigvals.max())[0][0]
    sigx = np.sqrt(np.abs(eigvals[0]))
    sigy = np.sqrt(np.abs(eigvals[1]))
    xrot_rad = np.arctan2(eigvecs[1, ind_of_max_eigvalue], eigvecs[0, ind_of_max_eigvalue])
    xrot_deg = (180 / np.pi) * xrot_rad
    if xrot_deg < 0: xrot_deg += 180
    if sigx < sigy:
        tmp = sigx
        sigx = sigy
        sigy = tmp
    return sigx, sigy, xrot_deg


def gauss2d(data, sigma, xrot=None, h=None, k=None, filled=0.):
    """
    Apply a 2 dimensional Gaussian for KDE.

    Note, the smoothing is done by applying a series of 1-D convolutions.

    Parameters
    ----------
    data : array_like
        The data over which to apply the gaussian filter

    sigma : float or sequence of floats
        The sigma values of the gaussian. If sequence, first element is sigma
        used to smooth the first axis. If single sigma provided, it is used
        for both axes.

    xrot : float, optional
        The value used to rotate the smoothing Gaussian

    h : int, optional
        The x-dimensional offset used to shift the resulting smoot grid

    k : int, optional
        The y-dimensional offset used to shift the resulting smooth grid

    filled : float, optional
        If data is masked, the masked elements will be treated by using the
        filled value.

    Returns
    -------
    data1 : array_like
        A smoothed version of the input array, data

    """
    if isinstance(data, np.ma.core.MaskedArray):
        masked = True
        mask = data.mask
        data = data.filled(filled)
    else:
        masked = False

    # If a rotated gaussian, provide correcting rotation since axes are
    # rotated in ndimage packages
    if xrot: xrot += 90

    # Get Sigmas for Gaussian:
    if getattr(sigma, '__iter__', False):
        sigx = sigma[0]
        try:
            sigy = sigma[1]
        except:
            sigy = sigx
    else:
        sigx = sigma
        sigy = sigma

    # If rotation needed, rotate grid to align for smoothing
    if xrot:
        data1 = ndimage.rotate(data, xrot, order=0, reshape=True)
    else:
        data1 = data

    # Apply Smoothing to grid
    data1 = ndimage.gaussian_filter(data1, [sigx, sigy], order=0,
                                    mode='constant')

    # Now rotate back to original postion
    if xrot:
        data1 = ndimage.rotate(data1, -xrot, order=3, reshape=False)
        xdiff = (data1.shape[0] - data.shape[0])
        ydiff = (data1.shape[1] - data.shape[1])
        hxdiff = xdiff / 2
        hydiff = ydiff / 2

        if xdiff % 2 and ydiff % 2:
            data1 = data1[hxdiff+1:-hxdiff, hydiff+1:-hydiff]
        elif xdiff % 2:
            data1 = data1[hxdiff+1:-hxdiff, hydiff:-hydiff]
        elif ydiff % 2:
            data1 = data1[hxdiff:-hxdiff, hydiff+1:-hydiff]
        else:
            data1 = data1[hxdiff:-hxdiff, hydiff:-hydiff]

    # Now apply shift (remember, axes rotated so h is y and k is x)
    if h and k:
        data1 = ndimage.shift(data1, (k, h), order=0, prefilter=False)
    elif h and not k:
        data1 = ndimage.shift(data1, (0, h), order=0, prefilter=False)
    elif k and not h:
        data1 = ndimage.shift(data1, (k, 0), order=0, prefilter=False)
    if masked:
        return np.ma.array(data1, dtype=data.dtype, mask=mask, fill_value=0)
    else:
        return data1