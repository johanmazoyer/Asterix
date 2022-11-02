from Asterix.optics import roundpupil


def test_roundpupil():
    dim = 10
    rad = int(dim/2)
    pupil = roundpupil(dim, rad, no_pixel=False)

    zeros_left = (pupil[0, 0] == 0) and (pupil[0, 1] == 0) and (pupil[0, 2] == 0)
    center = (pupil[0, 3] == 1) and (pupil[0, 4] == 1) and (pupil[0, 5] == 1) and (pupil[0, 6] == 1)
    zeros_right = (pupil[0, 7] == 0) and (pupil[0, 8] == 0) and (pupil[0, 9] == 0)
    assert zeros_left and center and zeros_right, f"{dim}-pixel pupil is not centered between pixels, in center of array."
