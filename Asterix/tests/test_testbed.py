import os
import numpy as np
from Asterix import Asterix_root
from Asterix.utils import get_data_dir, read_parameter_file
from Asterix.optics import Pupil, Coronagraph, DeformableMirror
from Asterix.optics import Testbed as Bench
# we renamed Testbed here because pytest automatically assumes this class is a
# test because of the prefix "test" in Testbed, which throws a warning


def test_def_thd():

    silence = True

    parameter_file_test = os.path.join(Asterix_root, 'tests', "param_file_tests.ini")
    test_dir = get_data_dir(datadir="asterix_test_dir")

    model_local_dir = os.path.join(test_dir, "Model_local")

    # Load configuration file
    config = read_parameter_file(parameter_file_test)

    # wrapped vortex coronagraph
    model_config = config["modelconfig"]
    dm_config = config["DMconfig"]
    corona_config = config["Coronaconfig"]
    dm_config['DM2_active'] = True

    # wrapped vortex coronagraph
    corona_config["corona_type"] = 'wrapped_vortex'

    # Create all optical elements of the THD
    entrance_pupil = Pupil(model_config,
                           PupType=model_config["filename_instr_pup"],
                           angle_rotation=model_config["entrance_pup_rotation"],
                           Model_local_dir=model_local_dir,
                           silence=silence)
    dm2 = DeformableMirror(model_config, dm_config, Name_DM="DM2", Model_local_dir=model_local_dir, silence=silence)
    corono = Coronagraph(model_config, corona_config, Model_local_dir=model_local_dir, silence=silence)

    # Concatenate into the full testbed optical system
    optical_bench = Bench([entrance_pupil, dm2, corono], ["entrancepupil", "DM2", "corono"], silence=silence)

    testbed_psf = optical_bench.todetector_intensity()
    assert np.max(testbed_psf) < 5e-9, "PSF after wrapped vortex without aberrration should be better than 5e-9"

    assert np.allclose(testbed_psf, np.transpose(testbed_psf), rtol=0, atol=1e-14,
                       equal_nan=True), "PSF after testbed with no aberrration is not symmetric (transpose PSF != PSF)"
