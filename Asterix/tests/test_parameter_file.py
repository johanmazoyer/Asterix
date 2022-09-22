import os
from configobj import ConfigObj
from validate import Validator

from Asterix.optical_systems import Asterix_root

def test_example_parameter_file():
    # Define where to load the example parameter file from.
    parameter_file_ex = Asterix_root + os.path.sep + "Example_param_file.ini"

    # Define where to load the template parameter file from.
    configspec_file = Asterix_root + os.path.sep + "Param_configspec.ini"

    # Create the ConfigObj object with file to be tested and validation file, and the validator object.
    config = ConfigObj(parameter_file_ex, configspec=configspec_file, default_encoding="utf8")
    vtor = Validator()

    # Check our imported example parameter file against the template.
    checks = config.validate(vtor, copy=True)

    assert checks is True, "Your example parameter file does not correspond to the template."
