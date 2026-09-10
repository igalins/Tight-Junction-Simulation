"""Tests for the standalone electrodiffusion equations in paracellular_transport.physics."""
import math

import pytest

from paracellular_transport.models import Compartment
from paracellular_transport.physics import (
    EMF,
    calculate_flux,
    calculate_ion_change,
    divalent_membrane_potential,
    goldmann_equation,
    nernst_equation,
    shared_voltage,
)


class TestNernstEquation:
    @pytest.mark.parametrize("c_start,c_end", [(0, 100), (100, 0), (-5, 100), (100, -5)])
    def test_raises_for_nonpositive_concentration(self, constants, temperature, c_start, c_end):
        with pytest.raises(ValueError):
            nernst_equation(constants.R, temperature, constants.F, 1, c_start, c_end, "Na")

    def test_symmetric_concentrations_give_zero_potential(self, constants, temperature):
        assert nernst_equation(constants.R, temperature, constants.F, 1, 100, 100, "Na") == pytest.approx(0.0)

    def test_sign_flips_with_valence(self, constants, temperature):
        cation = nernst_equation(constants.R, temperature, constants.F, 1, 150, 15, "Na")
        anion = nernst_equation(constants.R, temperature, constants.F, -1, 150, 15, "Cl")
        assert cation > 0
        assert anion < 0
        assert cation == pytest.approx(-anion)


class TestGoldmannEquation:
    def test_matches_manual_formula(self, constants, temperature):
        R, F, T = constants.R, constants.F, temperature
        P_Na, P_Cl = 10.0, 1.0
        Na_L, Cl_L, Na_R, Cl_R = 50.0, 45.0, 145.0, 105.0
        expected = (R * T / F) * math.log((P_Na * Na_L + P_Cl * Cl_R) / (P_Na * Na_R + P_Cl * Cl_L))
        actual = goldmann_equation(R, T, F, P_Na, P_Cl, Na_L, Cl_L, Na_R, Cl_R)
        assert actual == pytest.approx(expected)

    def test_symmetric_concentrations_give_zero_potential(self, constants, temperature):
        result = goldmann_equation(constants.R, temperature, constants.F, 10.0, 1.0, 100.0, 100.0, 100.0, 100.0)
        assert result == pytest.approx(0.0, abs=1e-12)


class TestDivalentMembranePotential:
    def test_symmetric_concentrations_give_zero_potential(self, constants, temperature):
        result = divalent_membrane_potential(
            3.0, 10.0, 1.0,
            1.0, 1.0,      # Mg: bl, ap (equal)
            100.0, 100.0,  # Na: bl, ap (equal)
            100.0, 100.0,  # Cl: bl, ap (equal)
            constants.R, temperature, constants.F,
        )
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_reduces_to_goldmann_equation_when_p_mg_is_zero(self, constants, temperature):
        """With P_Mg=0 the Mg terms drop out and this must match goldmann_equation exactly
        for the same Na/Cl inputs -- pins down the reversed (basolateral, apical) argument
        order documented on divalent_membrane_potential vs goldmann_equation's (apical,
        basolateral) order."""
        R, F, T = constants.R, constants.F, temperature
        P_Na, P_Cl = 10.0, 1.0
        Na_L, Cl_L, Na_R, Cl_R = 50.0, 45.0, 145.0, 105.0  # L=apical, R=basolateral

        goldman = goldmann_equation(R, T, F, P_Na, P_Cl, Na_L, Cl_L, Na_R, Cl_R)
        divalent = divalent_membrane_potential(
            0.0, P_Na, P_Cl,
            999.0, 888.0,  # Mg concentrations are irrelevant when P_Mg=0
            Na_R, Na_L,    # bl, ap
            Cl_R, Cl_L,    # bl, ap
            R, T, F,
        )
        assert divalent == pytest.approx(goldman)


class TestEMF:
    def test_sign_convention(self):
        # nernst_pot - equilibrium_pot: matches the "positive EMF pulls ions" convention
        # documented on EMF.
        assert EMF(equilibrium_pot=-0.070, nernst_pot=0.060) == pytest.approx(0.130)
        assert EMF(equilibrium_pot=0.060, nernst_pot=-0.070) == pytest.approx(-0.130)
        assert EMF(equilibrium_pot=0.05, nernst_pot=0.05) == pytest.approx(0.0)


class TestCalculateFlux:
    @pytest.fixture
    def compartments(self):
        start = Compartment('start', na_conc=150.0, cl_conc=0, mg_conc=0)
        end = Compartment('end', na_conc=15.0, cl_conc=0, mg_conc=0)
        return start, end

    def test_positive_drive_uses_start_concentration(self, compartments):
        start, end = compartments
        # emf * p * z > 0
        flux = calculate_flux(emf=0.01, p=10.0, z=1, start_comp=start, end_comp=end, ion_name='Na')
        assert flux == pytest.approx(0.01 * 10.0 * 1 * start.concentrations['Na'])

    def test_negative_drive_uses_end_concentration(self, compartments):
        start, end = compartments
        # emf * p * z < 0
        flux = calculate_flux(emf=-0.01, p=10.0, z=1, start_comp=start, end_comp=end, ion_name='Na')
        assert flux == pytest.approx(-0.01 * 10.0 * 1 * end.concentrations['Na'])

    def test_exact_zero_drive_uses_end_concentration(self, compartments):
        """emf * p * z == 0 falls into the else/end_comp branch -- the code comment
        'Threshold changed from 1 to 0!!!' flags this boundary as previously debugged."""
        start, end = compartments
        flux = calculate_flux(emf=0.0, p=10.0, z=1, start_comp=start, end_comp=end, ion_name='Na')
        assert flux == pytest.approx(0.0)


class TestCalculateIonChange:
    def test_arithmetic(self, constants):
        flux, vol, dt = 2.5e-9, 8e-20, 1e-4
        expected = flux * dt * vol * constants.Avogadro
        assert calculate_ion_change(flux, vol, dt) == pytest.approx(expected)

    def test_zero_flux_gives_zero_change(self):
        assert calculate_ion_change(0.0, 8e-20, 1e-4) == pytest.approx(0.0)

    def test_avogadro_is_overridable(self):
        assert calculate_ion_change(1.0, 1.0, 1.0, avogadro=2.0) == pytest.approx(2.0)


class TestSharedVoltage:
    def test_equal_potentials_returns_that_potential_regardless_of_resistance(self):
        assert shared_voltage(U1=0.02, U2=0.02, R1=14.0, R2=19.0) == pytest.approx(0.02)
        assert shared_voltage(U1=0.02, U2=0.02, R1=1.0, R2=1000.0) == pytest.approx(0.02)

    def test_matches_thevenin_formula(self):
        U1, U2, R1, R2 = 0.02, 0.01, 14.0, 19.0
        I_a = (U1 - U2) / (R1 + R2)
        expected = U1 - I_a * R1
        assert shared_voltage(U1, U2, R1, R2) == pytest.approx(expected)
