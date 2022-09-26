# pylint: disable=invalid-name
# pylint: disable=trailing-whitespace

import inspect
import copy
import numpy as np

import Asterix.optics.optical_systems as optsy
import Asterix.optics.deformable_mirror as deformable_mirror

class Testbed(optsy.OpticalSystem):
    """
    
    Initialize and describe the behavior of a testbed.
    This is a particular subclass of Optical System, because we do not know what is inside
    It can only be initialized by giving a list of Optical Systems and it will create a
    "testbed" with contains all the Optical Systems and associated EF_through functions and
    correct normlaization

    AUTHOR : Johan Mazoyer

    """

    def __init__(self, list_os, list_os_names):
        """
        This function allows you to concatenate OpticalSystem objects to create a testbed:
        parameter:
            list_os:        list of OpticalSystem instances
                            all the systems must have been defined with
                            the same modelconfig or it will send an error.
                            The list order is form the first optics system to the last in the
                            path of the light (so usually from entrance pupil to Lyot pupil)

            list_os_names:  list of string of the same size as list_os 
                            Name of the optical systems. 
                            Then can then be accessed inside the Testbed object by os_#i = Testbed.list_os_names[i]

        Returns
        ------
            testbed : an optical system which is the concatenation of all the optical systems

        """
        if len(list_os) != len(list_os_names):
            print("")
            raise Exception("list of systems and list of names need to be of the same size")

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
                raise Exception("list_os[" + str(num_optical_sys) + "] is not an optical system")

            if list_os[num_optical_sys].modelconfig != self.modelconfig:
                print("")
                raise Exception("All optical systems need to be defined with the same initial modelconfig!")

            # if the os is a DM we increase the number of DM counter and
            # store the number of act and its name

            for params in inspect.signature(list_os[num_optical_sys].EF_through).parameters:
                known_keywords.append(params)

            if isinstance(list_os[num_optical_sys], deformable_mirror.DeformableMirror):

                #this function is to replace the DMphase variable by a XXphase variable
                # where XX is the name of the DM
                list_os[num_optical_sys].EF_through = _swap_DMphase_name(
                    list_os[num_optical_sys].EF_through, list_os_names[num_optical_sys] + "phase")
                known_keywords.append(list_os_names[num_optical_sys] + "phase")

                if list_os[num_optical_sys].active == False:
                    # if the Dm is not active, we just add it to the testbed model
                    # but not to the EF_through function
                    vars(self)[list_os_names[num_optical_sys]] = list_os[num_optical_sys]
                    continue

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
        known_keywords.append('photon_noise')
        known_keywords.append('nb_photons')
        known_keywords.append('in_contrast')

        # we remove doubloons
        # known_keywords = list(set(known_keywords))
        known_keywords = list(dict.fromkeys(known_keywords))

        # We remove arguments we know are wrong
        if 'DMphase' in known_keywords:
            known_keywords.remove('DMphase')
        if self.number_DMs > 0:
            # there is at least a DM, we add voltage_vector as an authorize kw
            known_keywords.append('voltage_vector')
            self.EF_through = _control_testbed_with_voltages(self, self.EF_through)

        # to avoid mis-use we only use specific keywords.
        known_keywords.remove('kwargs')

        self.EF_through = _clean_EF_through(self.EF_through, known_keywords)

        #initialize the max and sum of PSFs for the normalization to contrast
        self.measure_normalization()

    def voltage_to_phases(self, actu_vect, einstein_sum=False):
        """
        Generate the phase applied on each DMs of the testbed from a given vector of
        actuator amplitude. I split theactu_vect and  then for each DM, it uses
        DM.voltage_to_phase (no s)

        Parameters
        ----------
        actu_vect : float or 1D array of size testbed.number_act
                    values of the amplitudes for each actuator and each DM
        einstein_sum : boolean. default false
                        Use numpy Einstein sum to sum the pushact[i]*actu_vect[i]
                        gives the same results as normal sum. Seems ot be faster for unique actuator
                        but slower for more complex phases

        Returns
        ------
            3D array of size [testbed.number_DMs, testbed.dim_overpad_pupil,testbed.dim_overpad_pupil]
            phase maps for each DMs by order of light path in the same unit as actu_vect * DM_pushact

        AUTHOR : Johan Mazoyer

        """
        DMphases = np.zeros((self.number_DMs, self.dim_overpad_pupil, self.dim_overpad_pupil))
        indice_acum_number_act = 0

        if isinstance(actu_vect, (int, float)):
            return np.zeros(self.number_DMs) + float(actu_vect)

        if len(actu_vect) != self.number_act:
            raise Exception("voltage vector must be 0 or array of dimension testbed.number_act," +
                            "sum of all DM.number_act")

        for i, DM_name in enumerate(self.name_of_DMs):

            DM = vars(self)[DM_name]  # type: deformable_mirror.DeformableMirror
            actu_vect_DM = actu_vect[indice_acum_number_act:indice_acum_number_act + DM.number_act]
            DMphases[i] = DM.voltage_to_phase(actu_vect_DM, einstein_sum=einstein_sum)

            indice_acum_number_act += DM.number_act

        return DMphases

    def basis_vector_to_act_vector(self, vector_basis_voltage):
        """
        transform a vector of voltages on the mode of a basis in a  vector of
        voltages of the actuators of the DMs of the system

        Parameters
        ----------
        vector_basis_voltage: 1D-array real : 
                        vector of voltages of size (total(basisDM sizes)) on the mode of the basis for all
                        DMs by order of the light path

        Returns
        ------
        vector_actuator_voltage: 1D-array real : 
                        vector of base coefficients for all actuators of the DMs by order of the light path
                        size (total(DM actuators))
        
        """

        indice_acum_basis_size = 0
        indice_acum_number_act = 0

        vector_actuator_voltage = np.zeros(self.number_act)
        for DM_name in self.name_of_DMs:

            # we access each DM object individually
            DM = vars(self)[DM_name]  # type: deformable_mirror.DeformableMirror

            # we extract the voltages for this one
            # this voltages are in the DM basis
            vector_basis_voltage_for_DM = vector_basis_voltage[indice_acum_basis_size:indice_acum_basis_size +
                                                               DM.basis_size]

            # we change to actuator basis
            vector_actu_voltage_for_DM = np.dot(np.transpose(DM.basis), vector_basis_voltage_for_DM)

            # we recreate a voltages vector, but for each actuator
            vector_actuator_voltage[indice_acum_number_act:indice_acum_number_act +
                                    DM.number_act] = vector_actu_voltage_for_DM

            indice_acum_basis_size += DM.basis_size
            indice_acum_number_act += DM.number_act

        return vector_actuator_voltage


# Some internal functions to properly concatenate the EF_through functions
def _swap_DMphase_name(DM_EF_through_function, name_var):
    """
   A function to rename the DMphase parameter to another name (usually DMXXphase)
        
    AUTHOR : Johan Mazoyer

    Parameters:
    ------
        DM_EF_through_function : function
            the function of which we want to change the params
        name_var : string 
            the name of the  new name variable

    Returns
    ------
        the_new_function: function
            with name_var as a param

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
    A very small function to concatenate 2 functions
    AUTHOR : Johan Mazoyer

    Parameters:
    ------
        outer_fun: function
                x -> outer_fun(x)
        inner_fun: function 
                x -> inner_fun(x)

    Returns
        ------
        the concatenated function: function
                x -> outer_fun(inner_fun(x))

    """

    def new_EF_through_fun(**kwargs):

        new_kwargs_outer = copy.copy(kwargs)
        del new_kwargs_outer['entrance_EF']

        return outer_EF_through_fun(entrance_EF=inner_EF_through_fun(**kwargs), **new_kwargs_outer)

    return new_EF_through_fun


def _clean_EF_through(testbed_EF_through, known_keywords):
    """
    a functions to check that we do not set unknown keyword in
    the testbed EF through function. Maybe not necessary.

    AUTHOR : Johan Mazoyer

    Parameters:
    ------
         testbed_EF_through: function
         known_keywords: list of strings of known keywords

    Returns
    ------
        cleaned_testbed_EF_through: function
            a function where only known keywords are allowed
        
    """

    def wrapper(**kwargs):
        for passed_arg in kwargs.keys():
            if passed_arg == 'DMphase':
                raise Exception('DMphase is an ambiguous argument if you have several DMs.' +
                                ' Please use XXphase with XX = nameDM')
            if passed_arg not in known_keywords:
                raise Exception(passed_arg + 'is not a EF_through valid argument. Valid args are ' +
                                str(known_keywords))

        return testbed_EF_through(**kwargs)

    return wrapper


def _control_testbed_with_voltages(testbed: Testbed, testbed_EF_through):
    """
    A function to go from a testbed_EF_through with several DMXX_phase
    parameters (one for each DM), to a testbed_EF_through with a unique
    voltage_vector parameter of size testbed.number_act (or a single float, like 0.)

    the problem with DMXX_phase parameters is that it cannot be automated since it requires
    to know the name/number of the DMs in advance.

    DMXX_phase parameters can still be used, but are overridden by voltage_vector parameter
    if present.

    AUTHOR : Johan Mazoyer

    Parameters:
    ------
        DM_EF_through_function : function
                the function of which we want to change the params
        name_var : string 
                the name of the  new name variable

    Returns
    ------
        the_new_function: function
                with name_var as a param

    """

    def wrapper(**kwargs):
        if 'voltage_vector' in kwargs:
            voltage_vector = kwargs['voltage_vector']
            DM_phase = testbed.voltage_to_phases(voltage_vector)
            for i, DM_name in enumerate(testbed.name_of_DMs):
                name_phase = DM_name + "phase"
                kwargs[name_phase] = DM_phase[i]

        return testbed_EF_through(**kwargs)

    return wrapper
