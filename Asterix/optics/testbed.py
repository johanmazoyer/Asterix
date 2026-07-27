import inspect
import copy
import numpy as np

import Asterix.optics.optical_systems as optsy
import Asterix.optics.deformable_mirror as deformable_mirror


class Testbed(optsy.OpticalSystem):
    """Initialize and describe the behavior of a testbed. This is a particular
    subclass of Optical System, because we do not know what is inside It can
    only be initialized by giving a list of Optical Systems and it will create
    a "testbed" with contains all the Optical Systems and associated EF_through
    functions and correct normalization.

    AUTHOR : Johan Mazoyer
    """

    def __init__(self, list_os, list_os_names, silence=False):
        """This function allows you to concatenate OpticalSystem objects to
        create a testbed.

        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        list_os : list of OpticalSystem instances
            All the systems must have been defined with
            the same modelconfig or it will send an error.
            The list order is form the first optics system to the last in the
            path of the light (so usually from entrance pupil to Lyot pupil)
        list_os_names:  list of string of the same size as list_os
            Name of the optical systems.
            They can then be accessed inside the Testbed object by os_#i = Testbed.list_os_names[i]
        silence : boolean, default False.
            Whether to silence print outputs.

        Returns
        --------
        testbed : Asterix.optics.testbed.Tesbed
            An optical system which is the concatenation of all the optical systems
        """
        if len(list_os) != len(list_os_names):
            print("")
            raise ValueError("list of systems and list of names need to be of the same size")

        # Initialize the OpticalSystem class and inherit properties
        super().__init__(list_os[0].modelconfig)

        init_string = self.string_os

        # Initialize the EF_through_function
        self.EF_through = super().EF_through

        self.number_DMs = 0
        self.number_act = 0
        self.name_of_DMs = []

        # this is the collection of all the possible keywords that can be used in
        # practice in the final testbed.EF_through, so that can be used in
        # all the EF_through functions
        known_keywords = []

        # we store the name of all the sub systems
        self.subsystems = list_os_names

        # we concatenate the Optical Element starting by the end
        for num_optical_sys in range(len(list_os)):

            # we first check that all variables in the list are optical systems
            # defined the same way.
            if not isinstance(list_os[num_optical_sys], optsy.OpticalSystem):
                raise TypeError("list_os[" + str(num_optical_sys) + "] is not an optical system")

            if list_os[num_optical_sys].modelconfig != self.modelconfig:
                print("")
                raise ValueError("All optical systems need to be defined with the same initial modelconfig!")

            # if the os is a DM we increase the number of DM counter and
            # store the number of act and its name

            for params in inspect.signature(list_os[num_optical_sys].EF_through).parameters:
                known_keywords.append(params)

            if isinstance(list_os[num_optical_sys], deformable_mirror.DeformableMirror):

                # this function is to replace the DMphase variable by a XXphase variable
                # where XX is the name of the DM
                list_os[num_optical_sys].EF_through = _swap_DMphase_name(list_os[num_optical_sys].EF_through,
                                                                         list_os_names[num_optical_sys] + "phase")
                known_keywords.append(list_os_names[num_optical_sys] + "phase")

                if not list_os[num_optical_sys].active:
                    # if the Dm is not active, we just add it to the testbed model
                    # but not to the EF_through function
                    vars(self)[list_os_names[num_optical_sys]] = list_os[num_optical_sys]

                self.number_DMs += 1
                self.number_act += list_os[num_optical_sys].number_act
                self.name_of_DMs.append(list_os_names[num_optical_sys])

            # concatenation of the EF_through functions
            self.EF_through = _concat_fun(list_os[num_optical_sys].EF_through, self.EF_through)

            # we add all systems to the Optical System so that they can be accessed
            vars(self)[list_os_names[num_optical_sys]] = list_os[num_optical_sys]

            self.string_os += list_os[num_optical_sys].string_os.replace(init_string, '')

        # in case there is no coronagraph in the system, we still add
        # noFPM so that it does not break when we run transmission and max_sum_PSFs
        # which pass this keyword by default
        known_keywords.append('noFPM')
        known_keywords.append('nb_photons')
        known_keywords.append('in_contrast')

        # we remove doubloons
        # known_keywords = list(set(known_keywords))
        known_keywords = list(dict.fromkeys(known_keywords))

        # We remove arguments we know are wrong
        if 'DMphase' in known_keywords:
            known_keywords.remove('DMphase')
        if self.number_DMs > 0:
            # there is at least a DM, we add dm_command as an authorize kw
            known_keywords.append('dm_command')
            self.EF_through = _control_testbed_with_dmcommands(self, self.EF_through)

        # to avoid mis-use we only use specific keywords.
        known_keywords.remove('kwargs')

        self.EF_through = _clean_EF_through(self.EF_through, known_keywords)

        # initialize the max and sum of PSFs for the normalization to contrast
        self.measure_normalization()

    def dmcommands_to_phases(self, dm_command, einstein_sum=False):
        """Generate the phase applied on each DMs of the testbed from a given
        dm command for the testbed. I split the dm_command into individual DM dm_command
        then for each DM, it uses DM.dmcommand_to_phase (no s)

        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        dm_command : float or 1D array of size testbed.number_act
            Values of the amplitudes for each actuator and each DM in nm.
        einstein_sum : boolean. default False
            Use numpy Einstein sum to sum the pushact[i]*dm_command[i]
            gives the same results as normal sum. Seems ot be faster for unique actuator
            but slower for more complex phases.

        Returns
        --------
        phases : 3D array of size [testbed.number_DMs, testbed.dim_overpad_pupil,testbed.dim_overpad_pupil]
            Phase maps for each DMs by order of light path in radians.
        """
        DMphases = np.zeros((self.number_DMs, self.dim_overpad_pupil, self.dim_overpad_pupil))
        indice_acum_number_act = 0

        if isinstance(dm_command, (int, float)):
            return np.zeros(self.number_DMs) + float(dm_command)

        if len(dm_command) != self.number_act:
            raise ValueError("dm_command must be 0 or array of dimension testbed.number_act," +
                             "sum of all DM.number_act")

        for i, DM_name in enumerate(self.name_of_DMs):

            DM: deformable_mirror.DeformableMirror = vars(self)[DM_name]
            if DM.active:
                dm_command_DM = dm_command[indice_acum_number_act:indice_acum_number_act + DM.number_act]
                DMphases[i] = DM.dmcommand_to_phase(dm_command_DM, einstein_sum=einstein_sum)

            indice_acum_number_act += DM.number_act

        return DMphases

    def basis_vector_to_dmcommand(self, vector_basis_command):
        """transform a vector of command on the modes of a basis in a command
        of the DMs of the testbed.

        AUTHOR : Johan Mazoyer

        Parameters
        ----------
        vector_basis_command : 1D-array real
            Command of size (total(basisDM sizes)) on the mode of the basis.

        Returns
        --------
        dm_command : 1D-array real
            dm command in nm by order of the light path (size :total(DM actuators)).
        """

        indice_acum_basis_size = 0
        indice_acum_number_act = 0

        dm_command = np.zeros(self.number_act)
        for DM_name in self.name_of_DMs:

            # we access each DM object individually
            DM: deformable_mirror.DeformableMirror = vars(self)[DM_name]
            if DM.active:
                # we extract the command for this DM
                # this command are in the DM basis
                vector_basis_command_for_DM = vector_basis_command[indice_acum_basis_size:indice_acum_basis_size +
                                                                   DM.basis_size]

                # we change to the actuator basis
                command_for_DM = np.dot(np.transpose(DM.basis), vector_basis_command_for_DM)

                # we concatenate DM commands to obtain a single command for the testbed
                dm_command[indice_acum_number_act:indice_acum_number_act +
                                        DM.number_act] = command_for_DM

                indice_acum_basis_size += DM.basis_size
            indice_acum_number_act += DM.number_act

        return dm_command

    def indiv_DM_command_to_testbed_command(self, dmcommand_indiv, DM_name):
        """Transform a command on a single DM to command of all
        tesbted DMs using zeros on the other DMs.

        Parameters:
        --------
        dmcommand_indiv : 1D-array real
            the individual DM command in nm.
        DM_name : string
            The name of the DM you which to apply the command to.

        Returns
        --------
        testbed_command : 1D-array real of dim testbed.number_act
            the commands of the testbed DM with dmcommand_indiv at the
            position of DM DM_name and zero elsewhere.
        """

        if DM_name not in self.name_of_DMs:
            raise ValueError("DM_name must be in the list of DMs")

        testbed_command = np.zeros(self.number_act)
        indice_acum_number_act = 0
        for DM_name_here in self.name_of_DMs:

            # we access each DM object individually
            DM: deformable_mirror.DeformableMirror = vars(self)[DM_name_here]

            if DM_name_here == DM_name:
                if not DM.active:
                    raise ValueError("DM_name must be active to send commands.")

                if len(dmcommand_indiv) != DM.number_act:
                    raise ValueError(f"dmcommand_indiv must be of size the number_act of DM {DM_name} : {DM.number_act}")

                testbed_command[indice_acum_number_act:indice_acum_number_act + DM.number_act] = dmcommand_indiv
                return testbed_command
            else:
                indice_acum_number_act += DM.number_act

    def testbed_command_to_indiv_DM_command(self, testbed_command, DM_name):
        """Extract the commands of DM DM_name from a command of the full tesbted.

        Parameters:
        --------
        testbed_command : 1D-array of dim testbed.number_act
            the testbed command (all DM commands concatenated)
        DM_name : string
            The name of the DM to which you want to extract the individual command.

        Returns
        --------
        dmcommand_indiv : 1D-array real
            the individual DM command.

        """
        indice_acum_number_act = 0
        if DM_name not in self.name_of_DMs:
            raise ValueError("DM_name must be in the list of DMs")

        for DM_name_here in self.name_of_DMs:
            # we access each DM object individually
            DM: deformable_mirror.DeformableMirror = vars(self)[DM_name_here]

            if DM_name_here == DM_name:
                dmcommand_indiv = testbed_command[indice_acum_number_act:indice_acum_number_act + DM.number_act]
                return dmcommand_indiv

            else:
                indice_acum_number_act += DM.number_act


# Some internal functions to properly concatenate the EF_through functions
def _swap_DMphase_name(DM_EF_through_function, name_var):
    """A function to rename the DMphase parameter to another name (usually DMXXphase)

    AUTHOR : Johan Mazoyer

    Parameters:
    --------
    DM_EF_through_function : function
        The function of which we want to change the params.
    name_var : string
        The name of the  new name variable.

    Returns
    --------
    the_new_function : function
        Same function with name_var as a param.
    """

    def wrapper(**kwargs):

        if name_var not in kwargs.keys():
            kwargs[name_var] = 0.
        new_kwargs = copy.copy(kwargs)

        new_kwargs['DMphase'] = kwargs[name_var]

        return DM_EF_through_function(**new_kwargs)

    return wrapper


def _concat_fun(outer_EF_through_fun, inner_EF_through_fun):
    """
    A very small function to concatenate 2 functions.

    AUTHOR : Johan Mazoyer

    Parameters:
    --------
    outer_fun : function
        x -> outer_fun(x)
    inner_fun : function
        x -> inner_fun(x)

    Returns
    --------
    the concatenated function : function
        x -> outer_fun(inner_fun(x))

    """

    def new_EF_through_fun(**kwargs):

        new_kwargs_outer = copy.copy(kwargs)
        del new_kwargs_outer['entrance_EF']

        return outer_EF_through_fun(entrance_EF=inner_EF_through_fun(**kwargs), **new_kwargs_outer)

    return new_EF_through_fun


def _clean_EF_through(testbed_EF_through, known_keywords):
    """A functions to check that we do not set unknown keyword in the testbed
    EF through function. Maybe not necessary.

    AUTHOR : Johan Mazoyer

    Parameters:
    --------
    testbed_EF_through : function
    known_keywords: list of strings
        List of known keywords.

    Returns
    --------
    cleaned_testbed_EF_through : function
        A function where only known keywords are allowed.
    """

    def wrapper(**kwargs):
        for passed_arg in kwargs.keys():
            if passed_arg == 'DMphase':
                raise ValueError('DMphase is an ambiguous argument if you have several DMs.' +
                                 ' Please use XXphase with XX = nameDM')
            if passed_arg not in known_keywords:
                raise ValueError(passed_arg + 'is not a EF_through valid argument. Valid args are ' +
                                 str(known_keywords))

        return testbed_EF_through(**kwargs)

    return wrapper


def _control_testbed_with_dmcommands(testbed: Testbed, testbed_EF_through):
    """A function to go from a testbed_EF_through with several DMXX_phase
    parameters (one for each DM), to a testbed_EF_through with a unique
    dm_command parameter of size testbed.number_act (or a single float,
    like 0.)

    the problem with DMXX_phase parameters is that it cannot be automated since it requires
    to know the name/number of the DMs in advance.

    DMXX_phase parameters can still be used, but are overridden by dm_command parameter
    if present.

    AUTHOR : Johan Mazoyer

    Parameters:
    ------
    testbed : OpticalSystem.Testbed
        Testbed object which describes your testbed
    testbed_EF_through : function
        the EF_through function with DMXX_phase functions

    Returns
    --------
    the_new_function : function
        the EF_through function with dm_command as a parameters
    """

    def wrapper(**kwargs):
        if 'dm_command' in kwargs:
            dm_command = kwargs['dm_command']
            DM_phase = testbed.dmcommands_to_phases(dm_command)
            for i, DM_name in enumerate(testbed.name_of_DMs):
                name_phase = DM_name + "phase"
                kwargs[name_phase] = DM_phase[i]

        return testbed_EF_through(**kwargs)

    return wrapper
