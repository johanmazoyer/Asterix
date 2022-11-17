import os
from Asterix import Asterix_root
from Asterix.utils import get_data_dir, read_parameter_file
from Asterix.optics import Pupil, Coronagraph, DeformableMirror
from Asterix.optics import Testbed as Bench
# we rename testbed here because the pyhon test thinks it's a
# test because of the prefix "test" in testbec


def test_def_thd():

    parameter_file_path = os.path.join(Asterix_root, 'Example_param_file.ini')

    model_local_dir = os.path.join(get_data_dir(), "Model_local")

    # Load configuration file
    config = read_parameter_file(parameter_file_path)

    model_config = config["modelconfig"]
    dm_config = config["DMconfig"]
    corona_config = config["Coronaconfig"]
    dm_config['DM3_active'] = True

    # Create all optical elements of the THD
    entrance_pupil = Pupil(model_config,
                           PupType=model_config["filename_instr_pup"],
                           angle_rotation=model_config["entrance_pup_rotation"],
                           Model_local_dir=model_local_dir)
    dm3 = DeformableMirror(model_config, dm_config, Name_DM="DM3", Model_local_dir=model_local_dir)
    corono = Coronagraph(model_config, corona_config, Model_local_dir=model_local_dir)

    # Concatenate into the full testbed optical system
    optical_bench = Bench([entrance_pupil, dm3, corono], ["entrancepupil", "DM3", "corono"])

    optical_bench.todetector_intensity()
