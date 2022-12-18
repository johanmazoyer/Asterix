import platform
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from Asterix.utils import save_plane_in_fits


class MaskDH:
    """A very small class to do all the mask related stuff: retrieve parameters
    and combined them measure the mask and measure the string to save matrices.
    They're fast to measure so we do not need to save except in 'onbench' case.

    AUTHOR : Johan Mazoyer
    """

    def __init__(self, Correctionconfig):
        """initialize the mask object.

        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        Correctionconfig : dict
                general correction file which contains mask parameters
        """

        self.DH_shape = Correctionconfig["DH_shape"].lower()

        if self.DH_shape == "square":
            self.corner_pos = [float(i) for i in Correctionconfig["corner_pos"]]
        elif self.DH_shape == "circle":
            self.DH_side = Correctionconfig["DH_side"].lower()
            self.Sep_Min_Max = [float(i) for i in Correctionconfig["Sep_Min_Max"]]
            self.circ_offset = Correctionconfig["circ_offset"]
            self.circ_angle = Correctionconfig["circ_angle"]
        elif self.DH_shape in ["nodh", "custom"]:
            pass
        else:
            raise ValueError(f"'{self.DH_shape}' is not a valid DH shape.")

        self.string_mask = self.tostring()

    def creatingMaskDH(self, dimFP, FP_sampling, dir_save_all_planes=None, **kwargs):
        """Create a binary mask.

        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        dimFP : int
            size of the output FP mask
        FP_sampling : float
            resolution of focal plane pixel  per lambda / D
        dir_save_all_planes : string or None, default None.
            If not None, absolute directory to save all planes in fits for debugging purposes.
            This can generate a lot of fits especially if in a loop, use with caution.

        Returns
        --------
        maskDH : 2D array
            binary mask delimiting the DH
        """
        maskDH = np.ones((dimFP, dimFP))
        xx, yy = np.meshgrid(np.arange(dimFP) - dimFP / 2, np.arange(dimFP) - dimFP / 2)
        rr = np.hypot(yy, xx)

        if self.DH_shape == "nodh":
            return maskDH

        elif self.DH_shape == "custom":
            your_os = platform.system()
            img = Image.new(mode="L", size=(dimFP, dimFP))
            if your_os == "Darwin":
                fnt = ImageFont.truetype('/Library/Fonts/Arial Bold.ttf', size=int(dimFP / 4))
            elif your_os == "Linux":
                fnt = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeSerif.ttf', size=int(dimFP / 4))
            else:
                raise OSError("The path to fonts is not implemented for your OS yet.")
            d = ImageDraw.Draw(img)
            d.text((int(dimFP / 4), int(dimFP / 4)), "C N \nR S", font=fnt, fill=1)
            maskDH = np.flipud(np.asarray(img, dtype=float))

        elif self.DH_shape == "square":
            maskDH[xx < self.corner_pos[0] * FP_sampling] = 0
            maskDH[xx > self.corner_pos[1] * FP_sampling] = 0
            maskDH[yy < self.corner_pos[2] * FP_sampling] = 0
            maskDH[yy > self.corner_pos[3] * FP_sampling] = 0

        elif self.DH_shape == "circle":
            maskDH[rr >= self.Sep_Min_Max[1] * FP_sampling] = 0
            maskDH[rr < self.Sep_Min_Max[0] * FP_sampling] = 0
            if self.DH_side == "right":
                maskDH[xx < np.abs(self.circ_offset) * FP_sampling] = 0
                if self.circ_angle != 0:
                    maskDH[yy - xx / np.tan(self.circ_angle * np.pi / 180) > 0] = 0
                    maskDH[yy + xx / np.tan(self.circ_angle * np.pi / 180) < 0] = 0
            elif self.DH_side == "left":
                maskDH[xx > -np.abs(self.circ_offset) * FP_sampling] = 0
                if self.circ_angle != 0:
                    maskDH[yy - xx / np.tan(self.circ_angle * np.pi / 180) < 0] = 0
                    maskDH[yy + xx / np.tan(self.circ_angle * np.pi / 180) > 0] = 0
            elif self.DH_side == "bottom":
                maskDH[yy > -np.abs(self.circ_offset) * FP_sampling] = 0
                if self.circ_angle != 0:
                    maskDH[yy - xx * np.tan(self.circ_angle * np.pi / 180) > 0] = 0
                    maskDH[yy + xx * np.tan(self.circ_angle * np.pi / 180) > 0] = 0
            elif self.DH_side == "top":
                maskDH[yy < np.abs(self.circ_offset) * FP_sampling] = 0
                if self.circ_angle != 0:
                    maskDH[yy - xx * np.tan(self.circ_angle * np.pi / 180) < 0] = 0
                    maskDH[yy + xx * np.tan(self.circ_angle * np.pi / 180) < 0] = 0
            elif self.DH_side == "full":
                pass
            else:
                raise ValueError(
                    f"Circle DH_side can only be 'top', 'bottom', 'left', 'right' and 'full', not {self.DH_side}")

        else:
            raise ValueError(f"DH_shape can only be 'circle', 'square', 'nodh', 'custom', not {self.DH_shape}")

        if dir_save_all_planes is not None:
            name_plane = 'DH_mask'
            save_plane_in_fits(dir_save_all_planes, name_plane, maskDH)

        return maskDH

    def tostring(self):
        """create a mask String to be used to save .fits files. directly update
        self.stringdh.

        AUTHOR : Johan Mazoyer

        Returns
        --------
        stringdh : str
            mask String to save .fits files
        """

        if self.DH_shape == "square":
            stringdh = "MaskDH_square_" + "_".join(map(str, self.corner_pos))
        elif self.DH_shape == "circle":
            stringdh = "_circle_rad" + "_".join(map(str, self.Sep_Min_Max)) + "_" + str(self.DH_side)
            if self.DH_side != 'full':
                stringdh = stringdh + '_ang' + str(int(self.circ_angle))
                if self.circ_offset > 0.:
                    stringdh = stringdh + '_off' + str(self.circ_offset)
        elif self.DH_shape == "custom":
            stringdh = "CustomDHmask"
        elif self.DH_shape == "nodh":
            stringdh = "FullDH"

        return stringdh
