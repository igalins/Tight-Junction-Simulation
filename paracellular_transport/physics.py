"""Electrodiffusion equations for paracellular ion transport across a tight junction.

All functions here are pure (no shared state) and segment-agnostic: they operate
on plain numbers and ``Compartment`` objects, with no knowledge of any specific
nephron segment or scenario.
"""
import math

import numpy as np
from scipy.constants import Avogadro as _AVOGADRO


def nernst_equation(R, T, F, z, C_start, C_end, ion_name):
    """Equilibrium (Nernst) potential for one ion across a boundary.

    Args:
        R: Gas constant.
        T: Absolute temperature (K).
        F: Faraday constant.
        z: Ion valence (e.g. +1 for Na+, -1 for Cl-, +2 for Mg2+).
        C_start: Concentration on the starting (e.g. apical) side.
        C_end: Concentration on the ending (e.g. basolateral) side.
        ion_name: Ion label, used only for the error message.

    Returns:
        The Nernst potential in volts.

    Raises:
        ValueError: If either concentration is non-positive (the log is undefined).
    """
    if C_start <= 0 or C_end <= 0:
        raise ValueError(
            f"Non-positive concentration for {ion_name}: "
            f"C_start={C_start:.6f}, C_end={C_end:.6f}"
        )

    return ((R * T) / (F * z)) * math.log(C_start / C_end)


def goldmann_equation(R, T, F, P_Na, P_Cl, Na_L, Cl_L, Na_R, Cl_R):
    """Goldman-Hodgkin-Katz membrane potential for monovalent ions (Na+, Cl-) only.

    L/R follow the apical/basolateral convention used throughout this package:
    L = apical (left) side, R = basolateral (right) side. For Na+, a positive
    result means a positive flux (apical to basolateral); for Cl-, a positive
    result means a negative flux (basolateral to apical).

    Args:
        R: Gas constant.
        T: Absolute temperature (K).
        F: Faraday constant.
        P_Na: Na+ permeability.
        P_Cl: Cl- permeability.
        Na_L: Apical Na+ concentration.
        Cl_L: Apical Cl- concentration.
        Na_R: Basolateral Na+ concentration.
        Cl_R: Basolateral Cl- concentration.

    Returns:
        The membrane potential in volts.
    """
    epsilon = 1e-15  # avoids a division-by-zero when all permeabilities/concentrations are 0
    numerator = (P_Na * Na_L) + (P_Cl * Cl_R) + epsilon
    denominator = (P_Na * Na_R) + (P_Cl * Cl_L) + epsilon

    return (R * T / F) * math.log(numerator / denominator)


def divalent_membrane_potential(P_Mg, P_Na, P_Cl, a_Mg_bl, a_Mg_ap, a_Na_bl, a_Na_ap, a_Cl_bl, a_Cl_ap, R, T, F):
    """Membrane potential for a junction that also carries a divalent cation (Mg2+).

    Quadratic-equation extension of the Goldman equation to include Mg2+. Only
    the "+sqrt" root of the quadratic is physical here; the "-sqrt" root causes
    a math domain error for physiological inputs and is not used.

    Note the argument order is deliberately (basolateral, apical) per ion —
    the OPPOSITE of ``goldmann_equation``'s (apical, basolateral) order. This
    is existing, verified behavior; callers must transpose arguments
    accordingly (see ``Junctions.calculate_potentials``).

    Args:
        P_Mg: Mg2+ permeability.
        P_Na: Na+ permeability.
        P_Cl: Cl- permeability.
        a_Mg_bl: Basolateral Mg2+ concentration.
        a_Mg_ap: Apical Mg2+ concentration.
        a_Na_bl: Basolateral Na+ concentration.
        a_Na_ap: Apical Na+ concentration.
        a_Cl_bl: Basolateral Cl- concentration.
        a_Cl_ap: Apical Cl- concentration.
        R: Gas constant.
        T: Absolute temperature (K).
        F: Faraday constant.

    Returns:
        The membrane potential in volts.
    """
    u = math.log((np.sqrt((P_Na * a_Na_bl - P_Na * a_Na_ap - P_Cl * a_Cl_bl + P_Cl * a_Cl_ap)**2 - 4 * (4 * P_Mg * a_Mg_bl + P_Na * a_Na_bl + P_Cl * a_Cl_ap) *
                            (-4 * P_Mg * a_Mg_ap - P_Na * a_Na_ap - P_Cl * a_Cl_bl)) - P_Na * a_Na_bl + P_Na * a_Na_ap + P_Cl * a_Cl_bl - P_Cl * a_Cl_ap) /
                            (2 * (4 * P_Mg * a_Mg_bl + P_Na * a_Na_bl + P_Cl * a_Cl_ap)))
    membrane_pot = u * R * T / F  # in V
    return membrane_pot


def EMF(equilibrium_pot, nernst_pot):
    """Electromotive force driving an ion: the gap between its Nernst potential and the membrane potential.

    Args:
        equilibrium_pot: The membrane potential at the boundary (from a Goldman-family equation).
        nernst_pot: The ion's own Nernst (equilibrium) potential.

    Returns:
        The EMF in volts. E.g. if nernst_pot is +60 mV for Na+ and equilibrium_pot
        is -70 mV, this returns +130 mV: a strong pull on Na+ from apical to
        basolateral.
    """
    return nernst_pot - equilibrium_pot


def calculate_flux(emf, p, z, start_comp, end_comp, ion_name):
    """Ion flux across a junction, proportional to EMF, permeability, and donor-side concentration.

    The donor compartment is whichever side the net electrochemical drive
    (``emf * p * z``) pushes ions away from: ``start_comp`` when driving flux
    from start to end, ``end_comp`` otherwise (including the exact zero case).

    Args:
        emf: Electromotive force for this ion (see ``EMF``).
        p: Permeability of this ion at the junction.
        z: Ion valence.
        start_comp: The "start" (e.g. apical) ``Compartment``.
        end_comp: The "end" (e.g. basolateral) ``Compartment``.
        ion_name: Key into each compartment's ``concentrations`` dict.

    Returns:
        The flux (sign indicates direction: positive = start to end).
    """
    C_start = start_comp.concentrations[ion_name]
    C_end = end_comp.concentrations[ion_name]
    if emf * p * z > 0:  # ions move from apical to basolateral
        return emf * p * z * C_start
    else:  # ions move from basolateral to apical
        return emf * p * z * C_end


def calculate_ion_change(flux, vol_liters, dt, avogadro=_AVOGADRO):
    """Convert a flux into the number of ions gained/lost over one timestep.

    Args:
        flux: Ion flux, as returned by ``calculate_flux``.
        vol_liters: Compartment volume in liters.
        dt: Timestep duration (s).
        avogadro: Avogadro constant (defaults to ``scipy.constants.Avogadro``).

    Returns:
        The change in ion count for this timestep.
    """
    ion_change = flux * dt * vol_liters * avogadro
    return ion_change


def shared_voltage(U1, U2, R1, R2):
    """Combine two parallel junctions' potentials into one shared (Thevenin-equivalent) voltage.

    Used when two parallel paracellular pathways (e.g. different claudin
    isoforms) with different resistances connect the same two compartments:
    physically, they must share one voltage at that boundary, found here as
    the terminal voltage of the equivalent Thevenin circuit.

    Args:
        U1: Potential computed independently for pathway 1.
        U2: Potential computed independently for pathway 2.
        R1: Resistance of pathway 1.
        R2: Resistance of pathway 2.

    Returns:
        The shared terminal voltage ``U_k``.
    """
    I_a = (U1 - U2) / (R1 + R2)  # equalization current
    U_k = U1 - I_a * R1  # shared terminal voltage
    return U_k
