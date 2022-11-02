import os

from Asterix.utils import get_data_dir, read_parameter_file
from Asterix.optics import Pupil, Coronagraph, DeformableMirror, Testbed


class THD2(Testbed):
    """Testbed object configured for THD2, from input configfile.

    AUTHOR : ILa

    Attributes
    ----------
    config : dict
        A read-in .ini parameter file.
    """

    def __init__(self,
                 parameter_file,
                 new_model_config={},
                 new_dm_config={},
                 new_corona_config={},
                 ):
        """
        Parameters
        ----------
        parameter_file : string
            Absolute path to an .ini parameter file.
        new_model_config : dict, optional
            Can be used to directly change a parameter in the MODELconfig section of the input parameter file.
        new_dm_config : dict, optional
            Can be used to directly change a parameter in the DMconfig section of the input parameter file.
        new_corona_config : dict, optional
            Can be used to directly change a parameter in the Coronaconfig section of the input parameter file.
        """

        # Load configuration file
        self.config = read_parameter_file(parameter_file,
                                          NewMODELconfig=new_model_config,
                                          NewDMconfig=new_dm_config,
                                          NewCoronaconfig=new_corona_config)

        model_config = self.config["modelconfig"]
        dm_config = self.config["DMconfig"]
        corona_config = self.config["Coronaconfig"]
        model_local_dir = os.path.join(get_data_dir(config_in=self.config["Data_dir"]), "Model_local")

        # Create all optical elements of the THD
        entrance_pupil = Pupil(model_config,
                               PupType=model_config["filename_instr_pup"],
                               angle_rotation=model_config["entrance_pup_rotation"],
                               Model_local_dir=model_local_dir)
        dm1 = DeformableMirror(model_config, dm_config, Name_DM="DM1", Model_local_dir=model_local_dir)
        dm3 = DeformableMirror(model_config, dm_config, Name_DM="DM3", Model_local_dir=model_local_dir)
        corono = Coronagraph(model_config, corona_config, Model_local_dir=model_local_dir)

        # Concatenate into the full testbed optical system
        super().__init__([entrance_pupil, dm1, dm3, corono], ["entrancepupil", "DM1", "DM3", "corono"])
