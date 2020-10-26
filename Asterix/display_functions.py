import numpy as np
import matplotlib.pyplot as plt

def custom_plot(pup, img):
    """ --------------------------------------------------
    Plots two images next to each other.
    -------------------------------------------------- """
    f1 = plt.figure(1, figsize=(10, 5))
    f1.clf()
    ax1 = f1.add_subplot(121)
    ax1.imshow(pup, cmap="hot")
    ax2 = f1.add_subplot(122)
    ax2.imshow(img, cmap="hot")
    f1.tight_layout()


def four_plot(img1, img2, img3, img4):
    """ --------------------------------------------------
    Plots four images next to each other.
    -------------------------------------------------- """
    f1 = plt.figure(1, figsize=(22, 11))
    f1.clf()
    ax1 = f1.add_subplot(141)
    ax1.imshow(img1, cmap="hot")
    ax2 = f1.add_subplot(142)
    ax2.imshow(img2, cmap="hot")
    ax3 = f1.add_subplot(143)
    ax3.imshow(img3, cmap="hot")
    ax4 = f1.add_subplot(144)
    ax4.imshow(img4, cmap="hot")
    f1.tight_layout()


def determinecontrast(image, chiffre):
    """ --------------------------------------------------
    Determine contrast rms in one image, radially in the Dark Hole
    -------------------------------------------------- """
    xx, yy = np.meshgrid(np.arange(isz) - (isz) / 2, np.arange(isz) - (isz) / 2)
    rr = np.hypot(yy, xx)
    contrast = np.zeros(int(isz / 2 / chiffre))
    for i in chiffre * np.arange(int(isz / 2 / chiffre)):
        whereimage = np.zeros((isz, isz))
        whereimage[np.where(rr >= i)] = 1
        whereimage[np.where(rr >= i + chiffre)] = 0
        whereimage[np.where(xx <= 30)] = 0
        # whereimage[np.where(abs(yy)<10)]=0
        imagebis = (abs(image)) * whereimage
        contrast[int(i / chiffre)] = np.std(imagebis[np.where(whereimage != 0)])
    return contrast
