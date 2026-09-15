"""Compartment and junction objects that make up a paracellular transport chain."""
from .physics import (
    EMF,
    calculate_flux,
    calculate_ion_change,
    divalent_membrane_potential,
    goldmann_equation,
    nernst_equation,
)


class Compartment(object):
    """A fluid compartment (e.g. tubule lumen, tight-junction pocket, or blood side) with ion concentrations."""

    def __init__(self, name, na_conc, cl_conc, volume, mg_conc=0):
        """Create a compartment.

        Args:
            name: Compartment label (e.g. 'A', 'B').
            na_conc: Initial Na+ concentration.
            cl_conc: Initial Cl- concentration.
            volume: Compartment volume (liters).
            mg_conc: Initial Mg2+ concentration (default 0, for monovalent-only runs).
        """
        self.name = name
        self.concentrations = {'Na': na_conc, 'Cl': cl_conc, 'Mg': mg_conc}
        self.volume = volume

    def get_ion_counts(self, ion_conc, volume, avogadro):
        """Convert a concentration to an absolute ion count for this compartment's volume.

        Args:
            ion_conc: Ion concentration.
            volume: Compartment volume in liters.
            avogadro: Avogadro constant.

        Returns:
            Number of ions.
        """
        return ion_conc * volume * avogadro


class Junctions(object):
    """A tight junction connecting two compartments, with per-ion permeabilities.

    'Apical' and 'basolateral' name the two sides consistently with
    ``paracellular_transport.physics`` (apical = 'L'/left, basolateral =
    'R'/right in the Goldman-family equations).
    """

    def __init__(self, comp_apical: Compartment, comp_basolateral: Compartment, p_na, p_cl, p_mg, voltage_clamp: float = None):
        """Create a junction between two compartments.

        Args:
            comp_apical: The apical-side ``Compartment``.
            comp_basolateral: The basolateral-side ``Compartment``.
            p_na: Na+ permeability.
            p_cl: Cl- permeability.
            p_mg: Mg2+ permeability.
            voltage_clamp: If set, ``calculate_potentials`` returns this fixed
                value instead of computing one from concentrations — used to
                impose an externally-set (e.g. pump-driven) potential, or to
                temporarily fix this junction's contribution to a
                shared-voltage calculation (see ``engine.run_simulation``).
        """
        self.apical = comp_apical
        self.basolateral = comp_basolateral

        self.permeabilities = {'Na': p_na, 'Cl': p_cl, 'Mg': p_mg}

        self.voltage_clamp = voltage_clamp

    def calculate_potentials(self, R, T, F, divalent=False):
        """Compute (or return the clamped) membrane potential across this junction.

        Args:
            R: Gas constant.
            T: Absolute temperature (K).
            F: Faraday constant.
            divalent: If True, use the Mg2+-extended Goldman equation
                (``physics.divalent_membrane_potential``); otherwise use the
                monovalent-only Goldman equation (``physics.goldmann_equation``).

        Returns:
            The membrane potential in volts — ``self.voltage_clamp`` if set,
            otherwise computed from the two compartments' concentrations.
        """
        if self.voltage_clamp is not None:
            return self.voltage_clamp
        na_apical = self.apical.concentrations['Na']
        na_basolateral = self.basolateral.concentrations['Na']
        cl_apical = self.apical.concentrations['Cl']
        cl_basolateral = self.basolateral.concentrations['Cl']
        mg_apical = self.apical.concentrations['Mg']
        mg_basolateral = self.basolateral.concentrations['Mg']

        if divalent:
            # divalent_membrane_potential takes (basolateral, apical) per ion —
            # the opposite order of goldmann_equation below. See physics.py.
            membrane_pot = divalent_membrane_potential(self.permeabilities['Mg'], self.permeabilities['Na'], self.permeabilities['Cl'],
                                                       mg_basolateral, mg_apical, na_basolateral, na_apical, cl_basolateral, cl_apical, R, T, F)
        else:
            membrane_pot = goldmann_equation(R, T, F, self.permeabilities['Na'], self.permeabilities['Cl'],
                                            na_apical, cl_apical, na_basolateral, cl_basolateral)

        return membrane_pot

    def calculate_fluxes(self, R, T, F, ion_list, membrane_pot, dt):
        """Compute each ion's count change across this junction for one timestep.

        For each ion in ``ion_list``: derives its Nernst potential, combines
        it with ``membrane_pot`` into an EMF, computes the resulting flux, and
        converts that flux into an ion-count change using the donor side's
        own volume (whichever side ``calculate_flux`` drew its concentration
        from) — this is the ion count actually leaving that compartment, and
        conservation carries it unchanged to the other side; converting it
        back into each side's own concentration change is the caller's job
        (see ``engine.run_simulation``), since that requires both volumes.

        Note the valence lookup assumes any ion that isn't 'Na' or 'Cl' is a
        divalent cation (z=+2) — a known limitation if a future segment adds a
        second monovalent ion (e.g. K+) alongside Mg2+.

        Args:
            R: Gas constant.
            T: Absolute temperature (K).
            F: Faraday constant.
            ion_list: Ion keys to process, e.g. ('Na', 'Cl', 'Mg').
            membrane_pot: This junction's membrane potential (volts).
            dt: Timestep duration (s).

        Returns:
            Dict mapping each ion to its count change for this timestep.
        """
        ion_count_changes = {}

        for ion in ion_list:
            p = self.permeabilities[ion]
            if ion == 'Na':
                z = 1
            elif ion == 'Cl':
                z = -1
            else:
                z = 2

            conc_apical = self.apical.concentrations[ion]
            conc_basolateral = self.basolateral.concentrations[ion]

            # Nernst + Membrane Pot. -> EMF -> flux
            nernst = nernst_equation(R, T, F, z, conc_apical, conc_basolateral, ion)
            emf = EMF(membrane_pot, nernst)
            flux = calculate_flux(emf, p, z, self.apical, self.basolateral, ion)

            # Convert flux to an ion count using whichever side donated it
            # (calculate_flux's own apical/basolateral branch, mirrored via flux's sign).
            donor_volume = self.apical.volume if flux > 0 else self.basolateral.volume
            ion_count_changes[ion] = calculate_ion_change(flux, donor_volume, dt)

        return ion_count_changes
