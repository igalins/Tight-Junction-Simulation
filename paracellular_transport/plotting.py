"""Plotting helpers for simulation results.

All functions here take history/time_axis data (as produced by
``engine.run_simulation``) rather than a specific scenario, so they work
unchanged for any nephron segment. Every function that can save a figure
takes an optional ``save_path`` — nothing is written to disk unless the
caller explicitly asks for it.
"""
import matplotlib.pyplot as plt

ION_LABELS = {'Na': 'Na$^+$', 'Cl': 'Cl$^-$', 'Mg': 'Mg$^{2+}$'}


def plot_ion_dynamics(history, time_axis, save_path=None):
    """Plot Na+, Cl-, and Mg2+ concentrations for all compartments over time.

    Args:
        history: dict of compartment name -> ion -> list of concentrations,
            e.g. ``result.concentration_history``.
        time_axis: List of simulation times.
        save_path: If given, save the figure to this path.
    """
    fig, (ax_na, ax_cl, ax_mg) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    for comp_name, ions in history.items():
        ax_na.plot(time_axis, ions['Na'], label=f'Na+ {comp_name}')
    ax_na.set_ylabel('Na+ (mol/L)')
    ax_na.set_title('Sodium Concentration Profile')
    ax_na.legend(loc='center right', fontsize='small')
    ax_na.grid(True, alpha=0.3)

    for comp_name, ions in history.items():
        ax_cl.plot(time_axis, ions['Cl'], label=f'Cl- {comp_name}')
    ax_cl.set_ylabel('Cl- (mol/L)')
    ax_cl.set_title('Chloride Concentration Profile')
    ax_cl.legend(loc='center right', fontsize='small')
    ax_cl.grid(True, alpha=0.3)

    for comp_name, ions in history.items():
        ax_mg.plot(time_axis, ions['Mg'], label=f'Mg2+ {comp_name}')
    ax_mg.set_ylabel('Mg2+ (mol/L)')
    ax_mg.set_xlabel('Time (a.u.)')
    ax_mg.set_title('Magnesium Concentration Profile')
    ax_mg.legend(loc='center right', fontsize='small')
    ax_mg.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_flow_dynamics(flow_history, time_axis, title=None, save_path=None):
    """Plot the net movement (flux * dt) of ions between junctions, for one pathway.

    Args:
        flow_history: dict of junction key -> ion -> list of ion-count
            changes, e.g. ``result.flow_history['10b']``.
        time_axis: List of simulation times.
        title: Optional figure super-title.
        save_path: If given, save the figure to this path.
    """
    fig, (ax_na, ax_cl, ax_mg) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold")

    for j_name, ions in flow_history.items():
        ax_na.plot(time_axis, ions['Na'], label=f'Flow {j_name}')
    ax_na.set_ylabel('Na+ Net Flow (arbitrary unit)')
    ax_na.set_title('Sodium Flow Rate Between Compartments')
    ax_na.axhline(0, color='black', lw=1, ls='--')
    ax_na.legend(loc='upper right', fontsize='small')
    ax_na.grid(True, alpha=0.3)

    for j_name, ions in flow_history.items():
        ax_cl.plot(time_axis, ions['Cl'], label=f'Flow {j_name}')
    ax_cl.set_ylabel('Cl- Net Flow (arbitrary unit)')
    ax_cl.set_title('Chloride Flow Rate Between Compartments')
    ax_cl.axhline(0, color='black', lw=1, ls='--')
    ax_cl.legend(loc='upper right', fontsize='small')
    ax_cl.grid(True, alpha=0.3)

    for j_name, ions in flow_history.items():
        if len(ions['Mg']) == 0:
            continue
        ax_mg.plot(time_axis, ions['Mg'], label=f'Flow {j_name}')
    ax_mg.set_ylabel('Mg2+ Net Flow (arbitrary unit)')
    ax_mg.set_xlabel('Time (a.u.)')
    ax_mg.set_title('Magnesium Flow Rate Between Compartments')
    ax_mg.axhline(0, color='black', lw=1, ls='--')
    ax_mg.legend(loc='upper right', fontsize='small')
    ax_mg.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_membrane_potential(transepithelial_potential, time_axis, title=None, save_path=None):
    """Plot a summed transepithelial potential over time.

    Args:
        transepithelial_potential: List of potentials, e.g. from
            ``engine.transepithelial_potential``.
        time_axis: List of simulation times.
        title: Optional figure super-title.
        save_path: If given, save the figure to this path.
    """
    plt.figure(figsize=(12, 6))
    if title:
        plt.suptitle(title, fontsize=16, fontweight="bold")
    plt.plot(time_axis, transepithelial_potential)
    plt.ylabel('Transepithelial Potential (mV)')
    plt.xlabel('Time (a.u.)')
    plt.title('Total Transepithelial Potential (A→D)')
    plt.axhline(0, color='black', lw=1, ls='--')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def total_flow(flow_history, ion):
    """Sum one pathway's flux for one ion across all its junctions, at each timestep.

    Args:
        flow_history: dict of junction key -> ion -> list, e.g.
            ``result.flow_history['10b']``.
        ion: Ion key, e.g. 'Mg'.

    Returns:
        A list of the summed flux at each timestep.
    """
    n_steps = len(next(iter(flow_history.values()))[ion])
    return [sum(flow_history[j][ion][t] for j in flow_history) for t in range(n_steps)]


def total_flow_all_pathways(result, ion):
    """Sum flux for one ion across every pathway and every junction in a result.

    Args:
        result: An ``engine.SimulationResult``.
        ion: Ion key, e.g. 'Mg'.

    Returns:
        A list of the summed flux (across all pathways) at each timestep.
    """
    per_pathway = [total_flow(flow_history, ion) for flow_history in result.flow_history.values()]
    return [sum(values) for values in zip(*per_pathway)]


def plot_total_flux_comparison(result_parallel, result_avg, ion='Mg', save_path=None):
    """Compare total flux (summed across all pathways/junctions) between two simulation results.

    Typically used to compare a "parallel claudins" result against an
    "averaged claudin" result for the same scenario, but works for any two
    ``SimulationResult``s sharing a time axis.

    Args:
        result_parallel: The (usually multi-pathway) ``SimulationResult`` to plot as the solid line.
        result_avg: The (usually single-pathway) ``SimulationResult`` to plot as the dashed line.
        ion: Ion to compare ('Na', 'Cl', or 'Mg').
        save_path: If given, save the figure to this path.

    Returns:
        A tuple ``(flux_parallel, flux_avg)`` of the two summed-flux lists.
    """
    flux_parallel = total_flow_all_pathways(result_parallel, ion)
    flux_avg = total_flow_all_pathways(result_avg, ion)

    label = ION_LABELS.get(ion, ion)

    plt.figure(figsize=(10, 4))
    plt.plot(result_parallel.time_axis, flux_parallel, label='Parallel (multi-pathway)')
    plt.plot(result_avg.time_axis, flux_avg, label='Averaged', linestyle='--')
    plt.axhline(0, color='black', lw=1, ls='--')
    plt.ylabel(f'Total {label} Flow (arbitrary unit)')
    plt.xlabel('Time (a.u.)')
    plt.title(f'{label} Reabsorption: Parallel vs Averaged Pathways')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

    return flux_parallel, flux_avg


def plot_mg_comparison(result_parallel, result_avg, save_path=None):
    """Backwards-compatible alias for ``plot_total_flux_comparison(..., ion='Mg')``."""
    return plot_total_flux_comparison(result_parallel, result_avg, ion='Mg', save_path=save_path)
