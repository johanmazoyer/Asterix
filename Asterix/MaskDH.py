# pylint: disable=invalid-name

import numpy as np


class MaskDH:
    """ --------------------------------------------------
        A very small class to do all the mask related stuff: retrieve parameters and
        combined them measure the mask and measure the string to save matrices.
        There are so fast to measure that we do not save them but if you
        absolutely want to you can do it here.
    -------------------------------------------------- """

    def __init__(self, Correctionconfig):
        """ --------------------------------------------------
        initialize the mask object.

        Parameters:
        ----------
        Correctionconfig: general correction file which contains mask parameters

        Author : Johan Mazoyer
        -------------------------------------------------- """

        self.DH_shape = Correctionconfig["DH_shape"].lower()

        if self.DH_shape == "square":
            self.corner_pos = [
                float(i) for i in Correctionconfig["corner_pos"]
            ]
        elif self.DH_shape == "circle":
            self.DH_side = Correctionconfig["DH_side"].lower()
            self.Sep_Min_Max = [
                float(i) for i in Correctionconfig["Sep_Min_Max"]
            ]

            self.circ_offset = Correctionconfig["circ_offset"]
            self.circ_angle = Correctionconfig["circ_angle"]
        else:
            raise Exception("Not valid DH Shape")

        self.name_string = self.tostring()

    def creatingMaskDH(self, dimFP, FP_sampling):
        """ --------------------------------------------------
        Create a binary mask.

        Parameters:
        ----------
        dimFP: int, size of the output FP mask
        FP_sampling: float, resolution of focal plane pixel  per lambda / D

        Return:
        ------
        maskDH: 2D array, binary mask

        Author : Johan Mazoyer
        -------------------------------------------------- """

        maskDH = np.ones((dimFP, dimFP))

        xx, yy = np.meshgrid(
            np.arange(dimFP) - (dimFP) / 2,
            np.arange(dimFP) - (dimFP) / 2)
        rr = np.hypot(yy, xx)

        if self.DH_shape == "square":

            maskDH[xx < self.corner_pos[0] * FP_sampling] = 0
            maskDH[xx > self.corner_pos[1] * FP_sampling] = 0
            maskDH[yy < self.corner_pos[2] * FP_sampling] = 0
            maskDH[yy > self.corner_pos[3] * FP_sampling] = 0

        if self.DH_shape == "circle":
            maskDH[rr >= self.Sep_Min_Max[1] * FP_sampling] = 0
            maskDH[rr < self.Sep_Min_Max[0] * FP_sampling] = 0
            if self.DH_side == "right":
                maskDH[xx < np.abs(self.circ_offset) * FP_sampling] = 0
                if self.circ_angle != 0:
                    maskDH[yy -
                           xx / np.tan(self.circ_angle * np.pi / 180) > 0] = 0
                    maskDH[yy +
                           xx / np.tan(self.circ_angle * np.pi / 180) < 0] = 0
            if self.DH_side == "left":
                maskDH[xx > -np.abs(self.circ_offset) * FP_sampling] = 0
                if self.circ_angle != 0:
                    maskDH[yy -
                           xx / np.tan(self.circ_angle * np.pi / 180) < 0] = 0
                    maskDH[yy +
                           xx / np.tan(self.circ_angle * np.pi / 180) > 0] = 0
            if self.DH_side == "bottom":
                maskDH[yy > -np.abs(self.circ_offset) * FP_sampling] = 0
                if self.circ_angle != 0:
                    maskDH[yy -
                           xx * np.tan(self.circ_angle * np.pi / 180) > 0] = 0
                    maskDH[yy +
                           xx * np.tan(self.circ_angle * np.pi / 180) > 0] = 0
            if self.DH_side == "top":
                maskDH[yy < np.abs(self.circ_offset) * FP_sampling] = 0
                if self.circ_angle != 0:
                    maskDH[yy -
                        xx * np.tan(self.circ_angle * np.pi / 180) < 0] = 0
                    maskDH[yy +
                        xx * np.tan(self.circ_angle * np.pi / 180) < 0] = 0
        return maskDH

    def tostring(self):
        """ create a mask String to be used to save .fits files
         Author : Johan Mazoyer
        """

        if self.DH_shape == "square":
            stringdh = "MaskDH_square_[" + "_".join(map(str,
                                                        self.corner_pos)) + "]"
        if self.DH_shape == "circle":
            stringdh = "_circle_rad[" + "_".join(map(
                str, self.Sep_Min_Max)) + "]_" + str(self.DH_side)
            if self.DH_side != 'full':
                stringdh = stringdh + '_ang' + str(int(self.circ_angle))
                if self.circ_offset > 0.:
                    stringdh = stringdh + '_off' + str(self.circ_offset)

        return stringdh
