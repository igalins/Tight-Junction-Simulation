"""Simulation of paracellular ion transport across kidney tubule tight junctions.

Segment-agnostic engine (``engine``) + physics (``physics``) + model objects
(``models``) + plotting (``plotting``), driven by ``Scenario`` configs
(``config``). Currently only Thick Ascending Limb (TAL) scenarios are
defined; adding a new nephron segment means adding new ``Scenario``
instances to ``config.py``, not changing the engine.
"""
from .config import (
    PHYSICAL_CONSTANTS,
    TAL_STATE1,
    TAL_STATE2_AVG,
    TAL_STATE2_PARALLEL,
    CompartmentSpec,
    JunctionSpec,
    Pathway,
    PhysicalConstants,
    Scenario,
    SimulationSettings,
)
from .engine import SimulationResult, build_scenario, run_scenario, run_simulation, transepithelial_potential
from .models import Compartment, Junctions
from .physics import (
    EMF,
    calculate_flux,
    calculate_ion_change,
    divalent_membrane_potential,
    goldmann_equation,
    nernst_equation,
    shared_voltage,
)
from .plotting import (
    plot_flow_dynamics,
    plot_ion_dynamics,
    plot_membrane_potential,
    plot_mg_comparison,
    plot_total_flux_comparison,
    total_flow,
    total_flow_all_pathways,
)

__all__ = [
    "PHYSICAL_CONSTANTS",
    "TAL_STATE1",
    "TAL_STATE2_AVG",
    "TAL_STATE2_PARALLEL",
    "CompartmentSpec",
    "JunctionSpec",
    "Pathway",
    "PhysicalConstants",
    "Scenario",
    "SimulationSettings",
    "SimulationResult",
    "build_scenario",
    "run_scenario",
    "run_simulation",
    "transepithelial_potential",
    "Compartment",
    "Junctions",
    "EMF",
    "calculate_flux",
    "calculate_ion_change",
    "divalent_membrane_potential",
    "goldmann_equation",
    "nernst_equation",
    "shared_voltage",
    "plot_flow_dynamics",
    "plot_ion_dynamics",
    "plot_membrane_potential",
    "plot_mg_comparison",
    "plot_total_flux_comparison",
    "total_flow",
    "total_flow_all_pathways",
]
