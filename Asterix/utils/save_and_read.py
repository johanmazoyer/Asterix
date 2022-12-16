import errno
import sys
import os
import time
import subprocess
import warnings

import datetime
import numpy as np

from astropy.io import fits
from configobj import ConfigObj
from validate import Validator

from Asterix import Asterix_root


def save_plane_in_fits(dir_save_fits, name_plane, image):
    """Function to quickly save a real or complex file in fits.

    Parameters
    ----------
    dir_save_fits: string
        path to directory to save the fits
    name_plane : string
        name of the plane.
        final name is
        - current_time_str + '_' + name_plane + '_RE_and_IM.fits' if complex
        - current_time_str + '_' + name_plane + '_RE.fits' if real
    image : numpy array
        to save. Can be of any dimension
    """
    current_time_str = datetime.datetime.today().strftime('%H_%M_%S_%f')[:-3]
    name_fits = current_time_str + '_' + name_plane

    if not os.path.exists(dir_save_fits):
        raise FileNotFoundError("Please define an existing directory for dir_save_fits keyword or None")

    # sometimes the image can be a single float (0 for phase or 1 for EF).
    if isinstance(image, (int, float)):
        print(name_plane + " is a constant, not save in fits")
        return

    if np.iscomplexobj(image):
        tofits_array = np.zeros((2, ) + image.shape)
        tofits_array[0] = np.real(image)
        tofits_array[1] = np.imag(image)
        fits.writeto(os.path.join(dir_save_fits, name_fits + '_RE_and_IM.fits'), tofits_array, overwrite=True)
    else:
        fits.writeto(os.path.join(dir_save_fits, name_fits + '_RE.fits'), image, overwrite=True)


def quickfits(tab, folder='', name='tmp'):
    """Quickly save an image to a fits file.

    By default, it will save on the desktop with a random name to avoid overwriting.
    Not sure if the default saving on Desktop works for Windows, but it works fine on MacOS and Linux.

    AUTHOR: Johan Mazoyer

    Parameters
    ----------
    tab : ndarray
        Image to be saved to disk.
    folder : string
        Path to save file location.
    name : string
        File name of images to save to disk.
    """

    if folder == '':
        desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
        bureau = os.path.join(os.path.join(os.path.expanduser('~')), 'Bureau')
        if os.path.exists(desktop):
            folder = desktop
        elif os.path.exists(bureau):
            # of you are french are you ?
            folder = bureau
        else:
            raise FileNotFoundError("I cannot find your desktop, please give me a folder to save the fits file to.")

    if name == 'tmp':
        current_time_str = datetime.datetime.today().strftime('_%H_%M_%S_%f')[:-3]
        name = name + current_time_str
    fits.writeto(os.path.join(folder, f'{name}.fits'), tab, overwrite=True)


def progress(count, total, status=''):
    """print a progress bar for a for loop.

    Parameters
    ----------
    count: int
        counter in the for loop

    total: int
        number of iterations in the for loop
    """
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 0)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()


def read_parameter_file(parameter_file,
                        NewMODELconfig={},
                        NewDMconfig={},
                        NewCoronaconfig={},
                        NewEstimationconfig={},
                        NewCorrectionconfig={},
                        NewLoopconfig={},
                        NewSIMUconfig={}):
    """Check existence of the given parameter file, read it and check validity.

    AUTHOR: Johan Mazoyer

    Parameters
    ----------
    parameter_file: string
        Absolute path to a .ini parameter file.
    NewMODELconfig: dict, optional
        Can be used to directly change a parameter in the MODELconfig section of the input parameter file.
    NewDMconfig: dict, optional
        Can be used to directly change a parameter in the DMconfig section of the input parameter file.
    NewCoronaconfig: dict, optional
        Can be used to directly change a parameter in the Coronaconfig section of the input parameter file.
    NewEstimationconfig: dict, optional
        Can be used to directly change a parameter in the Estimationconfig section of the input parameter file.
    NewCorrectionconfig: dict, optional
        Can be used to directly change a parameter in the Correctionconfig section of the input parameter file.
    NewSIMUconfig: dict, optional
        Can be used to directly change a parameter in the SIMUconfig section of the input parameter file.

    Returns
    --------
    config: dict
        Parameter dictionary
    """

    if not os.path.exists(parameter_file):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), parameter_file)

    configspec_file = os.path.join(Asterix_root, "Param_configspec.ini")

    if not os.path.exists(configspec_file):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), configspec_file)

    # Define configuration file
    config = ConfigObj(parameter_file, configspec=configspec_file, default_encoding="utf8")

    config["modelconfig"].update(NewMODELconfig)
    config["DMconfig"].update(NewDMconfig)
    config["Coronaconfig"].update(NewCoronaconfig)
    config["Estimationconfig"].update(NewEstimationconfig)
    config["Correctionconfig"].update(NewCorrectionconfig)
    config["Loopconfig"].update(NewLoopconfig)
    config["SIMUconfig"].update(NewSIMUconfig)

    test_validity_params = config.validate(Validator(), copy=True)

    if not test_validity_params:
        for name, section in test_validity_params.items():
            if section:
                continue
            for key, value in section.items():
                if not value:
                    raise ValueError(f'In section [{name}], parameter "{key}" is not properly defined')

    return config


def from_param_to_header(config):
    """
    Convert ConfigObj parameters to fits header type list
    AUTHOR: Axel Potier

    Parameters
    ----------
    config: dict
        config obj

    Returns
    --------
    header : dict
        list of parameters

    """
    header = fits.Header()
    for sect in config.sections:
        # print(config[str(sect)])
        for scalar in config[str(sect)].scalars:
            header[str(scalar)[:8]] = str(config[str(sect)][str(scalar)])
    return header


def get_data_dir(env_var_name="ASTERIX_DATA_PATH", config_in=None, datadir="asterix_data"):
    """Create a path to the local data directory.

    If the environment variable `env_var_name` exists, this is returned as the full data path and the input 'datadir'
        is ignored. You can set this individually on your OS.
    If the environment variable does not exist (default for all new users) but the user adapted the ini file and
        accesses the 'Data_dir' entry, the configfile entry is returned and 'datadir' is ignored.
    If the environment variable does not exist and the Data_dir entry in the ini file is not passed or set to '.',
        the directory 'datadir' is appended to the user's home directory and returned as an absolute path. On MacOS of
        user 'myuser' for example, this would return: '/Users/myuser/asterix_data'

    Parameters
    ----------
    env_var_name : string, optional
        Environment variable for optional override, default 'ASTERIX_DATA_PATH'.
    config_in : string, optional
        Directory name passed through from configuration file.
    datadir : string, optional
        Name of the top-level data directory, default "asterix_data".

    Returns
    ---------
    Absolute path to top-level data directory.
    """
    try:
        ret_path = os.environ[env_var_name]
        print(f"Using the following data path from env var {env_var_name}: '{ret_path}'")
        return ret_path
    except KeyError:
        pass

    if config_in is not None and config_in != '.':
        return config_in
    else:
        home_path = os.path.abspath(os.path.join(os.path.expanduser("~"), datadir))
        if not os.path.isdir(home_path):
            os.mkdir(home_path)
        return home_path


def create_experiment_dir(append=''):
    """Create the name for an experiment directory.

    Create a timestamp including current year, month and day, as well as hour, minute and second. Add the passed
    suffix 'append' before returning it.

    Parameters
    ----------
    append : string
        Filename suffix to add to timestamp, default is ''.
    """
    time_stamp = time.time()
    date_time_string = datetime.datetime.fromtimestamp(time_stamp).strftime("%Y%m%d_%H-%M-%S")

    if append != "":
        append = "_" + append

    experiment_folder = date_time_string + append
    return experiment_folder


def get_git_description():
    """ Return git description of current branch and commit. """

    # Ensure we run this in the code repo, regardless of current working dir
    codedir = os.path.dirname(__file__)

    def get_output_wrapper(cmd):
        return subprocess.check_output(cmd.split(), universal_newlines=True, cwd=codedir).strip()

    try:
        desc = get_output_wrapper('git rev-parse --short HEAD')
    except Exception:
        warnings.warn("Unable to get git description")
        desc = 'unable to get git hash'

    return desc
