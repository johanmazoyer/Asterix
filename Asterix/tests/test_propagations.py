import numpy as np
from Asterix.optics import butterworth_circle, fqpm_mask, mft, prop_fpm_regional_sampling, roundpupil


def test_mft_centering():
    """
    Test default centering of MFT, which in Asterix is defined to be from in between pixels to in between pixels.
    """
    pdim = 8
    rad = pdim / 2
    pup = roundpupil(pdim, rad)

    samp = 4
    efield = mft(pup, real_dim_input=pdim, dim_output=pdim, nbres=samp)
    img = np.abs(efield)**2

    four_equal_pixels = (img[3, 3] == img[3, 4]) and (img[3, 3] == img[4, 3]) and (img[4, 3]
                                                                                   == img[4, 3]) and (img[4, 4]
                                                                                                      == img[4, 4])
    assert four_equal_pixels, "PSF from MFT is not symmetric in four center pixels."


def test_butterworth():
    siz = 100
    rad = int(siz / 2)

    bfilter1 = butterworth_circle(siz, rad, order=1, xshift=0, yshift=-0.5)
    bfilter3 = butterworth_circle(siz, rad, order=3, xshift=-0.5, yshift=0)
    bfilter5 = butterworth_circle(siz, rad, order=5, xshift=0, yshift=0)
    bfilter15 = butterworth_circle(siz, rad, order=15, xshift=-0.5, yshift=-0.5)

    # Sample a couple of points on each curve
    assert np.isclose(bfilter1[rad][5], 0.48979608156473575, atol=1e-12)
    assert np.isclose(bfilter1[rad][46], 0.9903417466743302, atol=1e-12)
    assert np.isclose(bfilter1[rad][78], 0.6594378183081357, atol=1e-12)

    assert np.isclose(bfilter3[rad][7], 0.19279741283979013, atol=1e-12)
    assert np.isclose(bfilter3[rad][33], 0.9538423456242743, atol=1e-12)
    assert np.isclose(bfilter3[rad][58], 0.9994572552018458, atol=1e-12)

    assert np.isclose(bfilter5[rad][9], 0.0839930579026251, atol=1e-12)
    assert np.isclose(bfilter5[rad][51], 0.9999999999999949, atol=1e-12)
    assert np.isclose(bfilter5[rad][87], 0.1394527062547814, atol=1e-12)

    assert np.isclose(bfilter15[rad][6], 0.00024622432504434573, atol=1e-12)
    assert np.isclose(bfilter15[rad][35], 0.9999999593040473, atol=1e-12)
    assert np.isclose(bfilter15[rad][47], 1.0, atol=1e-12)
    assert np.isclose(bfilter15[rad][59], 0.999999999999871, atol=1e-12)
    assert np.isclose(bfilter15[rad][71], 0.9945811821280887, atol=1e-12)
    assert np.isclose(bfilter15[rad][89], 0.0010462165718708913, atol=1e-12)

    # Test general shape of curve
    assert bfilter15[rad][6] < bfilter15[rad][35]
    assert bfilter15[rad][35] < bfilter15[rad][47]
    assert bfilter15[rad][47] > bfilter15[rad][59]
    assert bfilter15[rad][59] > bfilter15[rad][71]
    assert bfilter15[rad][71] > bfilter15[rad][89]


def test_prop_area_sampling():
    dim = 512
    rad = dim / 2
    samp_outer = 4
    nbres_direct = dim / samp_outer

    pup = roundpupil(dim, rad, grey_pup_bin_factor=10)
    lyot_stop = roundpupil(dim, rad * 0.95, grey_pup_bin_factor=1)
    fpm = fqpm_mask(dim)

    direct_ef = mft(pup * lyot_stop, real_dim_input=dim, dim_output=dim, nbres=nbres_direct)
    direct_psf = np.abs(direct_ef)**2
    norm = direct_psf.max()

    # Propagation with different sampling in FPM areas
    res_list = np.array([0.1, 1, 10, 100])
    pre_ls_areas = prop_fpm_regional_sampling(pup, np.exp(1j * fpm), nbres=res_list, samp_outer=samp_outer)
    post_ls_areas = pre_ls_areas * lyot_stop

    coro_ef_areas = mft(post_ls_areas, real_dim_input=dim, dim_output=dim, nbres=nbres_direct)
    coro_psf_areas = np.abs(coro_ef_areas)**2 / norm

    # Uniform-sampling propagation
    pre_fpm = mft(pup, real_dim_input=dim, dim_output=dim, nbres=nbres_direct)
    post_fpm = pre_fpm * np.exp(1j * fpm)
    pre_ls_uniform = mft(post_fpm, real_dim_input=dim, dim_output=dim, nbres=nbres_direct, inverse=True)
    post_ls_uniform = pre_ls_uniform * lyot_stop

    coro_ef_uniform = mft(post_ls_uniform, real_dim_input=dim, dim_output=dim, nbres=nbres_direct)
    coro_psf_uniform = np.abs(coro_ef_uniform)**2 / norm

    # Comparison
    assert np.sum(np.abs(post_ls_areas)**2) < np.sum(np.abs(post_ls_uniform)**2)
    assert (np.max(coro_psf_areas) / np.max(coro_psf_uniform)) < 3e-2
