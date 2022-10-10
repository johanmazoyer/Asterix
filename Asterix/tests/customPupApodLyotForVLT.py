import numpy as np
import scipy.ndimage.interpolation as interp

def sphereApodizerRadialProfile(x):
    """ compute SPHERE APLC apodizer radial profile
        x = 0 at the center of the pupil and x = 1 on the outer edge

    Args :
        x (float or np.array) [fraction of radius] : normalized radius
    """
    a = 0.16329229667014913
    b = 4.789900916663095
    c = -11.928993634750901
    d = 7.510133546534877
    e = -1.0284249458007801
    f = 0.8227342681613615
    return a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x + f

def makeVLTpup(pupdiam, cobs, t_spiders, pupangle, spiders = True):
    """ Return VLT pup, based on make_VLT function from shesha/shesha/util/make_pupil.py

    Args :
        pupdiam (float) [pixel] : pupil diameter
        
        cobs (float) [fraction of diameter] : central obtruction diameter

        t_spiders (float) [fraction of diameter] : spider diameter

        pupangle (float) [deg] : pupil rotation angle

        spiders (bool, optional) : if False, return the VLT pupil without spiders
    """
    range = (0.5 * (1) - 0.25 / pupdiam)
    X = np.tile(np.linspace(-range, range, pupdiam, dtype=np.float32), (pupdiam, 1))
    R = np.sqrt(X**2 + (X.T)**2)
    pup = ((R < 0.5) & (R > (cobs / 2))).astype(np.float32)

    if spiders:
        angle = 50.5 * np.pi / 180.  # --> 50.5 degre *2 d'angle entre les spiders
        spiders_map = (
                (X.T >
                (X - cobs / 2 + t_spiders / np.sin(angle)) * np.tan(angle)) +
                (X.T < (X - cobs / 2) * np.tan(angle))) * (X > 0) * (X.T > 0)
        spiders_map += np.fliplr(spiders_map)
        spiders_map += np.flipud(spiders_map)
        spiders_map = interp.rotate(spiders_map, pupangle, order=0, reshape=False)

        pup = pup * spiders_map  
    return pup

def makeSphereApodizer(pupdiam, cobs, radialProfile = sphereApodizerRadialProfile):
    """ Return the SPHERE APLC apodizer.

    Args :
        pupdiam (float) [pixel] : pupil diameter
        
        cobs (float) [fraction of diameter] : central obtruction diameter

        radialProfile (function, optional) : apodizer radial transmission. Default is SPHERE APLC apodizer.
    """
    # creating VLT pup without spiders
    pup = makeVLTpup(pupdiam, cobs, t_spiders = 0, pupangle = 0, spiders = False)

    # applying apodizer radial profile
    X = np.tile(np.linspace(-1, 1, pupdiam, dtype=np.float32), (pupdiam, 1))
    R = np.sqrt(X**2 + (X.T)**2)
    apodizer = pup*radialProfile(R)
    return apodizer

def makeSphereLyotStop(pupdiam, cobs, t_spiders, pupangle, addCentralObs = 2*14/384, addSpiderObs = 2*5.5/384, lyotOuterEdgeObs = 7/384):
    """ Return the SPHERE Lyot stop

    Args:
        pupdiam (float) [pixel] : pupil diameter
        
        cobs (float) [fraction of diameter] : central obtruction diameter

        t_spiders (float) [fraction of diameter] : spider diameter

        pupangle (float) [deg] : pupil rotation angle

        addCentralObs (float) [fraction of diameter] : additional diameter of central obstruction
        
        addSpiderObs (float) [fraction of diameter] : additional diameter of spiders obstruction
        
        lyotOuterEdgeObs (float) [fraction of diameter] : outer edge obstruction size
    """
    lyotCentralObs   = cobs + addCentralObs

    range = 0.5
    X = np.tile(np.linspace(-range, range, pupdiam, dtype=np.float32), (pupdiam, 1))
    R = np.sqrt(X**2 + (X.T)**2)
    lyotCentralMap = ((R < 0.5) & (R > (lyotCentralObs / 2))).astype(np.float32)

    angle = 50.5 * np.pi / 180.  # --> 50.5 degre *2 d'angle entre les spiders
    lyotSpidersMap = (
            (X.T > (X - cobs / 2 + (t_spiders + addSpiderObs / 2) / np.sin(angle)) * np.tan(angle)) +
            (X.T < (X - cobs / 2 - addSpiderObs / 2 / np.sin(angle)) * np.tan(angle))
            ) * (X > 0) * (X.T > 0)
    lyotSpidersMap += np.fliplr(lyotSpidersMap)
    lyotSpidersMap += np.flipud(lyotSpidersMap)
    lyotSpidersMap = interp.rotate(lyotSpidersMap, pupangle, order=0, reshape=False)

    X = np.tile(np.linspace(-range, range, pupdiam, dtype=np.float32), (pupdiam, 1))
    R = np.sqrt(X**2 + (X.T)**2)
    lyotOuterEdge = (R < 0.5 - lyotOuterEdgeObs)
    
    return lyotCentralMap*lyotSpidersMap*lyotOuterEdge