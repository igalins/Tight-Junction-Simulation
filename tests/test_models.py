"""Tests for Compartment and Junctions, especially calculate_potentials and calculate_fluxes."""
import pytest

from paracellular_transport.models import Compartment, Junctions
from paracellular_transport.physics import EMF, calculate_flux, calculate_ion_change, divalent_membrane_potential, goldmann_equation, nernst_equation


class TestCompartment:
    def test_default_mg_conc_is_zero(self):
        comp = Compartment('A', na_conc=150.0, cl_conc=120.0, volume=8e-20)
        assert comp.concentrations == {'Na': 150.0, 'Cl': 120.0, 'Mg': 0}

    def test_get_ion_counts_arithmetic(self):
        comp = Compartment('A', na_conc=150.0, cl_conc=120.0, volume=8e-20)
        assert comp.get_ion_counts(ion_conc=150.0, volume=8e-20, avogadro=6.022e23) == pytest.approx(150.0 * 8e-20 * 6.022e23)


class TestCalculatePotentials:
    @pytest.fixture
    def asymmetric_junction(self):
        apical = Compartment('A', na_conc=50.0, cl_conc=45.0, volume=8e-20, mg_conc=0.2)
        basolateral = Compartment('D', na_conc=145.0, cl_conc=105.0, volume=8e-20, mg_conc=0.5)
        return Junctions(apical, basolateral, p_na=10.0, p_cl=1.0, p_mg=3.0)

    def test_voltage_clamp_overrides_calculation_monovalent(self, asymmetric_junction, constants, temperature):
        asymmetric_junction.voltage_clamp = -0.005
        result = asymmetric_junction.calculate_potentials(constants.R, temperature, constants.F, divalent=False)
        assert result == pytest.approx(-0.005)

    def test_voltage_clamp_overrides_calculation_divalent(self, asymmetric_junction, constants, temperature):
        asymmetric_junction.voltage_clamp = -0.005
        result = asymmetric_junction.calculate_potentials(constants.R, temperature, constants.F, divalent=True)
        assert result == pytest.approx(-0.005)

    def test_symmetric_concentrations_give_zero_potential_monovalent(self, constants, temperature):
        apical = Compartment('A', na_conc=100.0, cl_conc=100.0, volume=8e-20, mg_conc=0.5)
        basolateral = Compartment('D', na_conc=100.0, cl_conc=100.0, volume=8e-20, mg_conc=0.5)
        junction = Junctions(apical, basolateral, p_na=10.0, p_cl=1.0, p_mg=3.0)
        result = junction.calculate_potentials(constants.R, temperature, constants.F, divalent=False)
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_symmetric_concentrations_give_zero_potential_divalent(self, constants, temperature):
        apical = Compartment('A', na_conc=100.0, cl_conc=100.0, volume=8e-20, mg_conc=0.5)
        basolateral = Compartment('D', na_conc=100.0, cl_conc=100.0, volume=8e-20, mg_conc=0.5)
        junction = Junctions(apical, basolateral, p_na=10.0, p_cl=1.0, p_mg=3.0)
        result = junction.calculate_potentials(constants.R, temperature, constants.F, divalent=True)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_monovalent_branch_dispatches_to_goldmann_equation_with_apical_basolateral_order(
        self, asymmetric_junction, constants, temperature
    ):
        R, F, T = constants.R, constants.F, temperature
        result = asymmetric_junction.calculate_potentials(R, T, F, divalent=False)
        expected = goldmann_equation(
            R, T, F, 10.0, 1.0,
            asymmetric_junction.apical.concentrations['Na'], asymmetric_junction.apical.concentrations['Cl'],
            asymmetric_junction.basolateral.concentrations['Na'], asymmetric_junction.basolateral.concentrations['Cl'],
        )
        assert result == pytest.approx(expected)

    def test_divalent_branch_dispatches_to_divalent_membrane_potential_with_reversed_order(
        self, asymmetric_junction, constants, temperature
    ):
        """calculate_potentials(divalent=True) must pass (basolateral, apical) per ion to
        divalent_membrane_potential -- the reverse of the (apical, basolateral) order used
        for the monovalent branch. This pins that wiring, not just the underlying formula."""

        R, F, T = constants.R, constants.F, temperature
        result = asymmetric_junction.calculate_potentials(R, T, F, divalent=True)
        a, b = asymmetric_junction.apical, asymmetric_junction.basolateral
        expected = divalent_membrane_potential(
            3.0, 10.0, 1.0,
            b.concentrations['Mg'], a.concentrations['Mg'],
            b.concentrations['Na'], a.concentrations['Na'],
            b.concentrations['Cl'], a.concentrations['Cl'],
            R, T, F,
        )
        assert result == pytest.approx(expected)


class TestCalculateFluxes:
    def test_symmetric_compartments_and_zero_potential_give_no_change(self, constants, temperature):
        apical = Compartment('A', na_conc=145.0, cl_conc=105.0, volume=8e-20, mg_conc=0.5)
        basolateral = Compartment('D', na_conc=145.0, cl_conc=105.0, volume=8e-20, mg_conc=0.5)
        junction = Junctions(apical, basolateral, p_na=10.0, p_cl=1.0, p_mg=3.0, voltage_clamp=0.0)

        changes = junction.calculate_fluxes(constants.R, temperature, constants.F, ('Na', 'Cl', 'Mg'), membrane_pot=0.0, dt=1e-4)

        assert changes == {'Na': pytest.approx(0.0), 'Cl': pytest.approx(0.0), 'Mg': pytest.approx(0.0)}

    def test_matches_manually_chained_physics_functions(self, constants, temperature):
        """calculate_fluxes must chain nernst_equation -> EMF -> calculate_flux ->
        calculate_ion_change per ion with the correct valence (Na=+1, Cl=-1, else +2) --
        recompute that chain independently and compare, for an asymmetric gradient."""

        apical = Compartment('A', na_conc=50.0, cl_conc=45.0, volume=8e-20, mg_conc=0.2)
        basolateral = Compartment('D', na_conc=145.0, cl_conc=105.0, volume=8e-20, mg_conc=0.5)
        R, F, T = constants.R, constants.F, temperature
        dt, volume = 1e-4, 8e-20
        membrane_pot = -0.02

        junction = Junctions(apical, basolateral, p_na=10.0, p_cl=1.0, p_mg=3.0)
        changes = junction.calculate_fluxes(R, T, F, ('Na', 'Cl', 'Mg'), membrane_pot, dt)

        for ion, (p, z) in {'Na': (10.0, 1), 'Cl': (1.0, -1), 'Mg': (3.0, 2)}.items():
            nernst = nernst_equation(R, T, F, z, apical.concentrations[ion], basolateral.concentrations[ion], ion)
            emf = EMF(membrane_pot, nernst)
            flux = calculate_flux(emf, p, z, apical, basolateral, ion)
            expected = calculate_ion_change(flux, volume, dt)
            assert changes[ion] == pytest.approx(expected)

    def test_flux_scales_linearly_with_permeability(self, constants, temperature):
        """With the membrane potential clamped (independent of permeability), doubling
        p_na must exactly double the Na ion-count change."""
        apical = Compartment('A', na_conc=50.0, cl_conc=45.0, volume=8e-20, mg_conc=0.2)
        basolateral = Compartment('D', na_conc=145.0, cl_conc=105.0, volume=8e-20, mg_conc=0.5)
        R, F, T = constants.R, constants.F, temperature

        junction_1x = Junctions(apical, basolateral, p_na=10.0, p_cl=1.0, p_mg=3.0, voltage_clamp=-0.02)
        junction_2x = Junctions(apical, basolateral, p_na=20.0, p_cl=1.0, p_mg=3.0, voltage_clamp=-0.02)

        changes_1x = junction_1x.calculate_fluxes(R, T, F, ('Na',), membrane_pot=-0.02, dt=1e-4)
        changes_2x = junction_2x.calculate_fluxes(R, T, F, ('Na',), membrane_pot=-0.02, dt=1e-4)

        assert changes_2x['Na'] == pytest.approx(2 * changes_1x['Na'])
