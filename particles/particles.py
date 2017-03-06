import numpy as np
from scipy import constants as const

__author__ = "Daniel Winklehner, Philip Weigel"
__doc__ = "Simple class to hold and calculate particle data like energy, mass, charge, etc."

# Initialize some global constants
amu = const.value("atomic mass constant energy equivalent in MeV")
echarge = const.value("elementary charge")
clight = const.value("speed of light in vacuum")


class IonSpecies:

    def __init__(self,
                 label="proton",
                 mass_mev=const.value('proton mass energy equivalent in MeV'),
                 a=1.007316,
                 z=None,
                 q=1.0,
                 energy_mev=1.0):

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
    ion = IonSpecies(label="4He2+", mass_mev=3727.379378, a=4.0026022, q=2.0, energy_mev=30.0)
    # ion = IonSpecies(label=r"$\mathrm{H}_2^+$", mass_mev=1876.634889, a=2.0147, q=1.0, energy_mev=30.0)
    print("Species {}".format(ion.label))
    print("Relativistic gamma = {}".format(ion.gamma))
    print("Relativistic beta = {}".format(ion.beta))
    print("Relativistic b-rho = {} T-m".format(ion.b_rho))
    print("B Field for 40 cm radius = {} T".format(ion.b_rho / 0.4))
