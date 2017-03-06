import numpy as np
from scipy import constants as const

__author__ = "Daniel Winklehner, Philip Weigel"
__doc__ = "Simple class to hold and calculate particle data like energy, mass, charge, etc."

# Initialize some global constants
amu = const.value("atomic mass constant energy equivalent in MeV")
echarge = const.value("elementary charge")
clight = const.value("speed of light in vacuum")

presets = {'proton': {'mass_mev': const.value('proton mass energy equivalent in MeV'),
                      'a': 1.007316,
                      'z': 1.0,
                      'q': 1.0},
           'H2_1+': {'mass_mev': 1876.634889,
                     'a': 2.0147,
                     'z': 2.0,
                     'q': 1.0},
           '4He_2+': {'mass_mev': 3727.379378,
                      'a': 4.0026022,
                      'z': 2.0,
                      'q': 2.0}}


class IonSpecies:

    def __init__(self,
                 label,
                 energy_mev,
                 mass_mev=None,
                 a=None,
                 z=None,
                 q=None,
                 debug=False):

        """
        Simple ion species class that holds data and can calculate some basic values like rigidity and emergy.
        :param label: Name of the species, can be one of the presets:
            'protons'
            'H2_1+'
            '4He_2+'
        if it is not a preset, the following four have to be defined as well:
        :param mass_mev:
        :param a:
        :param z:
        :param q:
        :param energy_mev:
        """

        # Check if user wants a preset ion species:
        if label in presets.keys():

            species = presets[label]

            mass_mev = species["mass_mev"]
            z = species["z"]
            a = species["a"]
            q = species["q"]

            if debug:
                print("Using preset ion species {}:".format(label))
                print("m_0 = {:.2f} MeV/c^2, q = {:.1f} e, E_kin = {:.2f} MeV". format(mass_mev, q, energy_mev))

        # If not, check for missing initial values
        else:

            init_values = [mass_mev, a, z, q]

            if None in init_values:

                print("Sorry, ion species {} was initialized with missing values ('None')!".format(label))
                print("mass_mev = {}, a = {}, z = {}, q = {}". format(mass_mev, a, z, q))

                exit(1)

            else:

                if debug:
                    print("User defined ion species {}:".format(label))
                    print("m_0 = {:.2f} MeV/c^2, q = {:.1f} e, E_kin = {:.2f} MeV".format(mass_mev, q, energy_mev))

        # Initialize values (default for a proton)
        self.label = label            # A label for this species
        self.mass_mev = mass_mev      # Rest Mass (MeV/c^2)
        self.a = a                    # Mass number A of the ion (amu)
        self.z = z                    # Proton number Z of the ion (unitless)
        self.energy_mev = energy_mev  # Initial kinetic energy (MeV/amu)
        self.q = q                    # charge state

        # Init other variables, calculate later
        self.gamma = 0.0
        self.beta = 0.0
        self.b_rho = 0.0
        self.mass_kg = 0.0

        # Calculate mass of the particle in kg
        self.mass_kg = self.mass_mev * echarge * 1.0e6 / clight**2.0

        self.calculate_from_energy_mev()

    @property
    def q_over_a(self):
        return self.q / self.a

    @property
    def total_kinetic_energy_mev(self):
        return self.energy_mev * self.a

    @property
    def total_kinetic_energy_ev(self):
        return self.energy_mev * self.a * 1.0e6

    @property
    def v_m_per_s(self):
        return self.beta * clight

    @property
    def v_cm_per_s(self):
        return self.beta * clight * 1.0e2

    def calculate_from_energy_mev(self, energy_mev=None):

        if energy_mev is not None:

            self.energy_mev = energy_mev

        # Calculate relativistic parameters
        self.gamma = self.energy_mev * self.a / self.mass_mev + 1.0
        self.beta = np.sqrt(1.0 - self.gamma**(-2.0))

        # Calculate B-rho of the particle
        self.b_rho = self.beta * self.gamma * self.mass_mev * 1.0e6 / (self.q * clight)

    def calculate_from_b_rho(self, b_rho=None):

        if b_rho is not None:

            self.b_rho = b_rho

        # Calculate relativistic parameters from b_rho
        betagamma = self.b_rho * self.q * clight * 1.0e-6 / self.mass_mev
        self.gamma = np.sqrt(betagamma**2.0 + 1.0)
        self.beta = betagamma / self.gamma

        # Calculate energy_mev from b_rho (cave: energy_mev is per nucleon)
        self.energy_mev = self.mass_mev * (self.gamma - 1.0) / self.a


if __name__ == '__main__':

    print("Testing IonSpecies class:")

    ion = IonSpecies(label="4He_2+", energy_mev=30.0)

    print("Species {}".format(ion.label))
    print("Energy = {} MeV/amu".format(ion.energy_mev))
    print("Relativistic gamma = {}".format(ion.gamma))
    print("Relativistic beta = {}".format(ion.beta))
    print("B-rho = {} T-m".format(ion.b_rho))
    print("B Field for 40 cm radius = {} T".format(ion.b_rho / 0.4))
