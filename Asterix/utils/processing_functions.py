# pylint: disable=invalid-name
# pylint: disable=trailing-whitespace

import numpy as np


def resizing(image, new):
    """
    Resample the focal plane image to create a 2D array with new dimensions

    - v1.0 2020 A. Potier
    - v2.0 19/03/21 J Mazoyer clean names + if image is real, result is real.
    - v3.0 05/2021 J Mazoyer Replacing currenly with standard scipy function zoom
    - v4.0 06/2022 J Mazoyer Replacing with the rebin and crop function following discussion with L. Mugnier
    - v5.0 08/2022 J Mazoyer Rename to resizing


    Parameters
    ----------
    image: 2D array
        input image
    new: int
        Size of the output image after resizing, in pixels

    Returns
    ------
    Gvector: 2D array
        image resampled into new dimensions

    """

    dimScience = len(image)
    dimEstim = new

    Estim_bin_factor = int(np.round(dimScience / dimEstim))

    # if the image was not orinigally a factor of Estim_bin_factor we crop a few raws
    slightly_crop_image = crop_or_pad_image(image, dimEstim * Estim_bin_factor)

    resized_image = resize_crop_bin(slightly_crop_image, dimEstim)

    return resized_image


def cropimage(img, ctr_x, ctr_y, newsizeimg):
    """
    Crop an image to create a 2D array with new dimensions
    AUTHOR: Axel Potier

    Parameters
    ----------
    img: 2D array
        input image, can be non squared
    ctr_x: float
        Center of the input image in the x direction around which you make the cut
    ctr_y: float
        Center of the input image in the y direction around which you make the cut
    
    newsizeimg: int
        Size of the new image

    Returns
    ------
    Gvector: 2D array
        squared image cropped into new dimensions

    """
    newimgs2 = newsizeimg / 2
    return img[int(ctr_x - newimgs2):int(ctr_x + newimgs2), int(ctr_y - newimgs2):int(ctr_y + newimgs2), ]


def crop_or_pad_image(image, dimout):
    """
    crop or padd with zero to a 2D image depeding: 
        - if dimout < dim : cropped image around pixel (dim/2,dim/2)
        - if dimout > dim : image around pixel (dim/2,dim/2) surrounded by 0

    AUTHOR: Raphael Galicher

    Parameters
    ----------
    image : 2D array (float, double or complex)
            dim x dim array

    dimout : int
         dimension of the output array

    Returns
    ------
    im_out : 2D array (float)
        resized image

    """
    if float(dimout) < image.shape[0]:
        im_out = np.zeros((image.shape[0], image.shape[1]), dtype=image.dtype)
        im_out = image[int((image.shape[0] - dimout) / 2):int((image.shape[0] + dimout) / 2),
                       int((image.shape[1] - dimout) / 2):int((image.shape[1] + dimout) / 2)]
    elif dimout > image.shape[0]:
        im_out = np.zeros((dimout, dimout), dtype=image.dtype)
        im_out[int((dimout - image.shape[0]) / 2):int((dimout + image.shape[0]) / 2),
               int((dimout - image.shape[1]) / 2):int((dimout + image.shape[1]) / 2)] = image
    else:
        im_out = image
    return im_out


def rebin(image, factor=4, center_on_pixel=False):
    """
    bin the image by a factor. The dimension dim MUST be divisible by factor
    or it will raise an error. It this is not the case, use function resize_crop_bin

    we add a center_on_pixel option . If false we shift the image to put 0 freq in the corner 
    before binning and then shift back.
    The goal is if you have a PSF center in between 4 pixel this property is conserved.
    If center_on_pixel= True there is no way to conserve this property unless we 
    are binning by odd number

    AUTHOR: Johan Mazoyer

    Parameters
    ----------
    image : 2D array (float, double or complex)
            dim1 x dim2 array with dim1 and dim2 are divisible by factor

    factor : int
         factor of bin init_image size / final_image size

    center_on_pixe :bool (optional, default: False). 
                If False the PSf is shifhted before bining

    Returns
    ------
    im_out : 2D array (float)
        resized image of size dim1 // 4 x dim2//4

    """

    dim1, dim2 = image.shape

    if (dim1 % factor != 0) or (dim2 % factor != 0):
        raise Exception("Image in Bin function must be divisible by factor of bin")

    shape = (dim1 // factor, factor, dim2 // factor, factor)

    if not center_on_pixel:
        return np.fft.fftshift(np.fft.fftshift(image).reshape(shape).mean(-1).mean(1))
    else:
        return image.reshape(shape).mean(-1).mean(1)


def resize_crop_bin(image, new_dim, center_on_pixel=False):
    """
    resize the imge by : 
        - cropping entrance image to nearest multiplicative number of new_dim
        - bin the image to new_dim size

    we add a center_on_pixel option . If false we shift the image to put 0 freq in the corner 
    before binning and then shift back.
    The goal is if you have a PSF center in between 4 pixel this property is conserved.
    If center_on_pixel= True there is no way to conserve this property unless we 
    are binning by odd number

    AUTHOR: Johan Mazoyer

    Parameters
    ----------
    image : 2D array (float, double or complex)
            dim1 x dim2 array with dim1 and dim2 are divisible by factor

    new_dim : int
         dimenstion of output image. new_dim must be samller than dim of the entrance image

    center_on_pixel :bool (optional, default: False). 
                If False the PSf is shifhted before bining

    Returns
    ------
    im_out : 2D array (float)
        resized image of size new_dim x new_dim

    """

    dim1, dim2 = image.shape

    if (dim1 < new_dim) or (dim2 < new_dim):
        raise Exception("new_dim must be samller than dimensions of the entrance image")

    # check closest multiplicative factor
    dim_smaller = min(dim1, dim2)
    factor = dim_smaller // new_dim

    # crop at the right size. Careful with the centering @TODO check
    return_image = cropimage(image, dim1 // 2, dim2 // 2, factor * new_dim)

    # Bin at the right size
    return_image = rebin(return_image, factor, center_on_pixel=center_on_pixel)

    return return_image


def ft_subpixel_shift(image, xshift, yshift, fourier=False, complex_image=False, norm="backward"):
    """
    This function returns an image shifted by a non-integer amount via a
    Fourier domain computation.

    (Based on subpixel_shift.pro from ONERA's IDL library by Laurent Mugnier)
    Renamed into ft_subpixel_shift to be clear on its purpose

    AUTHORS: L.Mugnier, M.Kourdourli, J. Mazoyer

    05/09/2022 : Introduction in asterix. Kourdourli version
    05/09/2022 : add complex_array param Mazoyer
    05/09/2022 : we invert xshift and yshift to be in agreement with np.roll (integer shift in numpy) Mazoyer
    06/09/2022 : added integer shift if we can Mazoyer
    06/09/2022 : works for non square array / non even dimensions array Mazoyer

    Parameters
    ----------
    image : 2D numpy array 
            intial image to be shifted
    xshift : float
            amount of desired shift in X direction.
    yshift : float
            amount of desired shift in Y direction.
    fourier : bool (optional, , default False) 
            if "True", then the input image is assumed to be already Fourier 
            transformed, i.e. the input is FFT^-1(image).
    complex_image : bool (optional, , default False)  
            if "False", then the output array will be
            assumed to be real. If you want to shift an complex array, use complex_image = True
    
    norm : string default 'backward'
            'backward', 'forward' or 'ortho'. this is the same paramter as in numpy.fft functions
            https://numpy.org/doc/stable/reference/routines.fft.html#module-numpy.fft
            if 'backward' no normalisation is done on MFT(inverse = False) and normalisation 1/N is done in MFT(inverse = True)
            if 'forward' 1/N normalisation is done on MFT(inverse = False) and no normalisation is done in MFT(inverse = True)
            if 'ortho' 1/sqrt(N) normalisation is done in both directions.
            Note that norm = 'ortho' allows you to conserve energy between a focal plane and pupil plane
            The default is 'backward' to be consistent with numpy.fft.fft2 and numpy.fft.ifft2
               
    Returns
    ------
    shifted_image : 2D numpy array
            shifted array with respect to the xshift and yshift used as input.
    """
    sz = np.shape(image)
    NP = sz[0]
    NL = sz[1]

    if (not fourier) and float(xshift).is_integer() and float(yshift).is_integer():
        return np.roll(image, (xshift, yshift), axis=(0, 1))

    if fourier:
        ft_image = image
    else:
        ft_image = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(image), norm=norm))

    xshift_odd = 0
    if NP % 2 == 1:
        xshift_odd = 1 / 2
    yshift_odd = 0
    if NL % 2 == 1:
        yshift_odd = 1 / 2

    x_ramp = np.outer(np.arange(NP) - NP / 2 + xshift_odd, np.ones(NL))
    y_ramp = np.outer(np.ones(NP), np.arange(NL) - NL / 2 + yshift_odd)

    # tilt describes the phase term in exp(i*phi) we will use to shift the image
    # by multiplying in the Fourier space and convolving in the direct space

    tilt = (-2 * np.pi / NP) * xshift * x_ramp + (-2 * np.pi / NL) * yshift * y_ramp
    # shift -> exp(i*phi)
    shift = np.cos(tilt) + 1j * np.sin(tilt)
    # inverse FFT to go back to the initial space
    shifted_image = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(ft_image * shift), norm=norm))

    # if the initial data is real, we take the real part
    if not complex_image:
        shifted_image = np.real(shifted_image)

    return shifted_image


def find_sizes_closest2factor(init_size_large, factor_zoomout, max_allowed_fft_size):
    """
    This function returns the best sizes (best_size_large, best_size_small) so that
    best_size_small / init_size_large are closest to factor_zoomout with 
    best_size_small and init_size_large even integers

    AUTHORS: J Mazoyer

    05/09/2022 : Introduction in Asterix
    
    Parameters
    ----------
    init_size_large : int
        inital size

    factor_dezoom : float
        factor to be zoomed out (final_image size / init_image size). factor_dezoom<1

    max_allowed_fft_size : int 
        the maximum size to check
    
    Returns
    ------
    dimensions : tupple of float of len 2
        best_size_large, best_size_small
    """
    best_size_large = init_size_large
    best_size_small = int(np.round(factor_zoomout * best_size_large))
    close_to_integer = np.abs(factor_zoomout * best_size_large - best_size_small)

    # we try to find the best size to padd our array so that new_size*factor_zoomout is a integer
    # we want at least 2 times the size of the initial array
    # we want the initial and final size to be even
    for i in range(max_allowed_fft_size // 2 - init_size_large // 2):
        size_here = factor_zoomout * (init_size_large + 2 * i)

        if np.abs(size_here - np.round(size_here)) == 0 and int(size_here) % 2 == 0:
            # we have a perfect size !
            best_size_large = int(init_size_large + 2 * i)
            best_size_small = int(factor_zoomout * best_size_large)
            break

        if np.abs(size_here - np.round(size_here)) < close_to_integer and int(size_here) % 2 == 0:
            # new best size
            close_to_integer = np.abs(size_here - np.round(size_here))
            best_size_large = int(init_size_large + 2 * i)
            best_size_small = int(factor_zoomout * best_size_large)

    return best_size_large, best_size_small


def ft_zoom_out(image, factor_zoomout, complex_image=False, max_allowed_fft_size=2000):
    """
    This function returns an image zoom out with Fourier domain computation. The array is padded
    until max_allowed_fft_size and takes the best size so that factor_zoomout*size_padded is the closest 
    to an integer. 

    BE CAREFUL WITH THE CENTERING, IT IS HARD TO FIND A GOOD RULE FOR ALL CASES (ODD OR EVEN DIMENSION IN OUTPUT AND INPUT)
    SO IT IS WHAT IT IS AND USERS ARE ENCOURAGED TO CHECK IF THIS IS WHAT THEY WANT

    AUTHORS: J Mazoyer

    05/09/2022 : Introduction in asterix

    Parameters
    ----------
    image : 2D numpy array
        inital array. Must be square 

    factor_dezoom : float
        factor to be zoomed out (final_image size / init_image size). factor_dezoom<1

    complex_image : bool(optional input, default False) 
            if this keyword is "False", then the output array will be
            assumed to be real. If you want to shift an complex array, use complex_image = True

    max_allowed_fft_size : int (optional input, default 2000)
        the maximum size of the first fft. If you increase, you might find a better match but it might take longer

        
    Returns
    ------
    zoomed_out_array : 2D numpy array
        zoomed out array

    """
    sz = np.shape(image)
    NP = sz[0]
    NL = sz[1]

    if NL == NP and isinstance(factor_zoomout, (float, int)):
        if factor_zoomout > 1:
            raise Exception("factor_zoomout must be <=1")
        # in that case we have the exact same size before and after in both directions
        best_size_largex, best_size_smallx = find_sizes_closest2factor(2 * NP, factor_zoomout,
                                                                       max_allowed_fft_size)
        best_size_largey = best_size_largex
        best_size_smally = best_size_smallx
        factor_zoomoutx = factor_zoomouty = factor_zoomout
    else:
        if isinstance(factor_zoomout, (float, int)):
            if factor_zoomout > 1:
                raise Exception("factor_zoomout must be <=1")
            # differnt size initially but same factor
            best_size_largex, best_size_smallx = find_sizes_closest2factor(2 * NP, factor_zoomout,
                                                                           max_allowed_fft_size)
            best_size_largey, best_size_smally = find_sizes_closest2factor(2 * NL, factor_zoomout,
                                                                           max_allowed_fft_size)
            factor_zoomoutx = factor_zoomouty = factor_zoomout
        else:
            # different factors
            factor_zoomoutx = factor_zoomout[0]
            factor_zoomouty = factor_zoomout[1]
            if factor_zoomoutx > 1 or factor_zoomouty > 1:
                raise Exception("factor_zoomout must be <=1")

            best_size_largex, best_size_smallx = find_sizes_closest2factor(2 * NP, factor_zoomout[0],
                                                                           max_allowed_fft_size)
            best_size_largey, best_size_smally = find_sizes_closest2factor(2 * NL, factor_zoomout[1],
                                                                           max_allowed_fft_size)
    # print(2*NP,factor_zoomoutx, best_size_largex, best_size_smallx, (best_size_smallx - factor_zoomoutx*best_size_largex)/best_size_smallx*100 )
    # print(2*NL,factor_zoomouty, best_size_largey, best_size_smally, (best_size_smally - factor_zoomouty*best_size_largey)/best_size_smally*100 )

    new_image = np.zeros((best_size_largex, best_size_largey), dtype=image.dtype)
    new_image[int((best_size_largex - image.shape[0]) / 2):int((best_size_largex + image.shape[0]) / 2),
              int((best_size_largey - image.shape[1]) / 2):int((best_size_largey + image.shape[1]) /
                                                               2)] = image

    ft_image = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(new_image)))

    ft_image_cropped = ft_image[int((ft_image.shape[0] - best_size_smallx) /
                                    2):int((ft_image.shape[0] + best_size_smallx) / 2),
                                int((ft_image.shape[1] - best_size_smally) /
                                    2):int((ft_image.shape[1] + best_size_smally) / 2)]

    smaller_image = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(ft_image_cropped)))

    smaller_image_cropped = smaller_image[
        int((smaller_image.shape[0] - int(np.ceil(factor_zoomoutx * NP))) /
            2):int((smaller_image.shape[0] + int(np.ceil(factor_zoomoutx * NP))) / 2),
        int((smaller_image.shape[1] - int(np.ceil(factor_zoomouty * NL))) /
            2):int((smaller_image.shape[1] + int(np.ceil(factor_zoomouty * NL))) / 2)]

    # if the initial data is real, we take the real part
    if not complex_image:
        smaller_image_cropped = np.real(smaller_image_cropped)

    return smaller_image_cropped
    