import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import scipy.sparse as sps
import config_JY as cfg

# ============================================================
# 1. GRID DEFINITION
# ============================================================


def create_non_uniform_grid(bed_properties):
    """
    Create a 1D non-uniform grid with ghost cells for a column model.

    Returns:
        column_grid (dict): grid properties including xCentres, xWalls, deltaZ, etc.
    """
    nx = 31  # Number of physical cells
    nGhost = 1  # Number of ghost cells on each end
    xmin, xmax = 0, bed_properties["bed_length"]  # Column length from bed properties

    dX_wide = 2 * (xmax - xmin) / (2 * nx - 9)
    dX_small = dX_wide / 4

    xGhost_start = xmin - dX_small * (np.flip(np.arange(nGhost)) + 0.5)
    xFirst = xmin + (np.arange(3) + 0.5) * dX_small
    xCentral = xmin + xFirst[-1] + (np.arange(nx - 6) + 1 / 8 + 0.5) * dX_wide
    xEnd = xmin + xCentral[-1] + (1 / 8 + 0.5) * dX_wide + dX_small * np.arange(3)
    xGhost_end = xmax + dX_small * (np.arange(nGhost) + 0.5)

    xCentres = np.concatenate((xGhost_start, xFirst, xCentral, xEnd, xGhost_end))

    xWalls_s = np.arange(-nGhost, 4) * dX_small
    xWalls_m = xWalls_s[-1] + np.arange(1, nx - 5) * dX_wide
    xWalls_e = xWalls_m[-1] + np.arange(1, 3 + nGhost + 1) * dX_small
    xWalls = np.concatenate((xWalls_s, xWalls_m, xWalls_e))
    deltaZ = xWalls[1 : nx + 2 * nGhost + 1] - xWalls[: nx + 2 * nGhost]

    return {
        "num_cells": nx,
        "nGhost": nGhost,
        "xCentres": xCentres,
        "xWalls": xWalls,
        "deltaZ": deltaZ,
    }


# ============================================================
# 2. EXTRAPOLATION FUNCTIONS
# ============================================================


def quadratic_extrapolation(x0, y0, x1, y1, x2, y2, x):
    """Quadratic extrapolation using Lagrange interpolation."""
    L_0 = (x - x1) * (x - x2) / ((x0 - x1) * (x0 - x2))
    L_1 = (x - x0) * (x - x2) / ((x1 - x0) * (x1 - x2))
    L_2 = (x - x0) * (x - x1) / ((x2 - x0) * (x2 - x1))
    return y0 * L_0 + y1 * L_1 + y2 * L_2


def quadratic_extrapolation_derivative(x0, y0, x1, y1, x2):
    """Extrapolate value y2 such that dy/dx = 0 at x2."""
    return (y1 * (x0 - x2) ** 2 - y0 * (x1 - x2) ** 2) / (
        (x0 - x1) * (x0 + x1 - 2 * x2)
    )


def quadratic_extrapolation_derivative_nonzero(x0, y0, x1, y1, x2, a, b):
    """
    Extrapolate value y2 such that dy/dx = a * (b - y2) at x2.
    Used for inlet/outlet Dirichlet or Robin BCs.
    """
    return (a * b + (y0 - y1) / (x0 - x1) + y0 / (-x0 + x2) + y1 / (-x1 + x2)) / (
        a + 1 / (-x0 + x2) + 1 / (-x1 + x2)
    )


def cubic_extrapolation(x0, y0, x1, y1, x2, y2, x3, y3, x):
    """
    Cubic extrapolation using Lagrange interpolation.

    Uses 4 known points (x0,y0), (x1,y1), (x2,y2), (x3,y3) to construct a cubic polynomial,
    then evaluates it at x4.

    Args:
        x0, y0: First known point
        x1, y1: Second known point
        x2, y2: Third known point
        x3, y3: Fourth known point
        x4: x-coordinate where we want to extrapolate y4

    Returns:
        y4: Extrapolated value at x4
    """
    L_0 = (x - x1) * (x - x2) * (x - x3) / ((x0 - x1) * (x0 - x2) * (x0 - x3))
    L_1 = (x - x0) * (x - x2) * (x - x3) / ((x1 - x0) * (x1 - x2) * (x1 - x3))
    L_2 = (x - x0) * (x - x1) * (x - x3) / ((x2 - x0) * (x2 - x1) * (x2 - x3))
    L_3 = (x - x0) * (x - x1) * (x - x2) / ((x3 - x0) * (x3 - x1) * (x3 - x2))

    return y0 * L_0 + y1 * L_1 + y2 * L_2 + y3 * L_3


def cubic_extrapolation_derivative(x0, y0, x1, y1, x2, y2, x):
    """
    Extrapolate value y4 using cubic Lagrangian interpolation such that dy/dx = 0 at x4.

    Uses 4 known points (x0,y0), (x1,y1), (x2,y2), (x3,y3) to construct a cubic polynomial,
    then finds y4 at x4 such that the derivative constraint dy/dx = 0 is satisfied.

    Args:
        x0, y0: First known point
        x1, y1: Second known point
        x2, y2: Third known point
        x3: x-coordinate where we want to extrapolate y3

    Returns:
        y3: Extrapolated value at x4
    """
    term1 = y2 * (x0 - x) * (x1 - x) / ((x0 - x2) * (-x1 + x2) * (x2 - x))
    term2 = y1 * (x0 - x) * (x2 - x) / ((x0 - x1) * (x1 - x2) * (x1 - x))
    term3 = y0 * (x1 - x) * (-x2 + x) / ((x0 - x1) * (x0 - x2) * (x0 - x))

    y3 = (term1 + term2 + term3) / (1 / (-x0 + x) + 1 / (-x1 + x) + 1 / (-x2 + x))
    return y3


def cubic_extrapolation_derivative_nonzero(x0, y0, x1, y1, x2, y2, x, a, b):
    """
    Extrapolate value y4 using cubic Lagrangian interpolation such that dy/dx = a * (b - y4) at x4.

    Uses 4 known points (x0,y0), (x1,y1), (x2,y2), (x3,y3) to construct a cubic polynomial,
    then finds y4 at x4 such that the derivative constraint dy/dx = a * (b - y4) is satisfied.

    Args:
        x0, y0: First known point
        x1, y1: Second known point
        x2, y2: Third known point
        x3: x-coordinate where we want to extrapolate y3
        a, b: Parameters for derivative constraint dy/dx = a * (b - y4)

    Returns:
        y4: Extrapolated value at x4
    """
    term1 = y2 * (x0 - x) * (x1 - x) / ((x0 - x2) * (-x1 + x2) * (x2 - x))
    term2 = y1 * (x0 - x) * (x2 - x) / ((x0 - x1) * (x1 - x2) * (x1 - x))
    term3 = y0 * (x1 - x) * (-x2 + x) / ((x0 - x1) * (x0 - x2) * (x0 - x))

    y3 = (a * b + term1 + term2 + term3) / (
        a + 1 / (-x0 + x) + 1 / (-x1 + x) + 1 / (-x2 + x)
    )

    return y3


def interpolator(time_series, input_func, new_time_points):
    import scipy.interpolate

    y_akima = scipy.interpolate.Akima1DInterpolator(
        time_series, input_func, method="makima"
    )

    def y_interp(points):
        return y_akima(points)

    return y_interp


# ============================================================
# 3. JACOBIAN SPARSITY PATTERN
# ============================================================


def build_jacobian(num_cells):
    n_vars = 8 * num_cells + 8 + 3  # P, T, Tw, y1,y2,y3,n1,n2 + F(8) + E(3)

    # Initialize sparsity as False
    S = np.zeros((n_vars, n_vars), dtype=int)

    # For each cell, mark dependencies
    for i in range(num_cells):
        idxP = i
        idxT = num_cells + i
        idxTw = 2 * num_cells + i
        idxY1 = 3 * num_cells + i
        idxY2 = 4 * num_cells + i
        idxY3 = 5 * num_cells + i
        idxN1 = 6 * num_cells + i
        idxN2 = 7 * num_cells + i

        # Example: dPdt[i] depends on P[i-1:i+1], T[i-1:i+1], Tw[i], Y1,Y2,Y3,N1,N2
        deps = [idxP, idxT, idxTw, idxY1, idxY2, idxY3, idxN1, idxN2]
        for j in deps:
            S[idxP, j] = 1  # P eqn depends on these

        # Similarly fill T, Tw, Y1,Y2,Y3,N1,N2 rows
        for j in deps:
            S[idxT, j] = 1
            S[idxTw, j] = 1
            S[idxY1, j] = 1
            S[idxY2, j] = 1
            S[idxY3, j] = 1
            S[idxN1, j] = 1
            S[idxN2, j] = 1

        # Add neighbor coupling (i-1, i+1) for convection/dispersion
        if i > 0:
            for row in [idxP, idxT, idxTw, idxY1, idxY2, idxY3]:
                for col in [idxP - 1, idxT - 1, idxY1 - 1, idxY2 - 1, idxY3 - 1]:
                    S[row, col] = 1
        if i < num_cells - 1:
            for row in [idxP, idxT, idxTw, idxY1, idxY2, idxY3]:
                for col in [idxP + 1, idxT + 1, idxY1 + 1, idxY2 + 1, idxY3 + 1]:
                    S[row, col] = 1

    # Tracking variables F, E depend on boundaries
    start_F = 8 * num_cells
    start_E = start_F + 8
    S[start_F : start_F + 8, :] = 1
    S[start_E : start_E + 3, :] = 1

    return sps.csr_matrix(S)


# ============================================================
# 4. PHYSICAL PROPERTY MODELS
# ============================================================
def pressure_ramp(t, stage, pressure_previous_stage, bed_properties):
    """
    Define pressure profiles for different process stages.
    Now uses bed_properties from the enclosing scope.
    """
    if stage == "adsorption":
        return bed_properties["ambient_pressure"]

    elif stage == "blowdown":
        initial_pressure = pressure_previous_stage
        final_pressure = bed_properties["vacuum_pressure"]
        tau = 0.11
        return final_pressure + (initial_pressure - final_pressure) * np.exp(-t / tau)

    elif stage == "heating":
        initial_pressure = pressure_previous_stage
        final_pressure = bed_properties["vacuum_pressure"]
        tau = 0.11
        return final_pressure + (initial_pressure - final_pressure) * np.exp(-t / tau)

    elif stage == "desorption":
        initial_pressure = pressure_previous_stage
        final_pressure = bed_properties["vacuum_pressure"]
        tau = 0.11
        return final_pressure

    elif stage == "pressurisation":
        initial_pressure = pressure_previous_stage
        final_pressure = bed_properties["ambient_pressure"]
        tau = 10
        return final_pressure - (final_pressure - initial_pressure) * np.exp(-t / tau)

    elif stage == "cooling":
        initial_pressure = pressure_previous_stage
        final_pressure = bed_properties["vacuum_pressure"]
        tau = 0.11
        return final_pressure


def pressure_ramp_2(t, stage, pressure_previous_step, bed_properties):
    tau = 0.11
    ambient_pressure = bed_properties["ambient_pressure"]
    vacuum_pressure = bed_properties["vacuum_pressure"]
    P = None
    dPdz = None
    if stage == "adsorption":
        P = ambient_pressure
        dPdz = None
    elif stage == "blowdown":
        P = None
        dPdz = tau * (vacuum_pressure - pressure_previous_step)
    elif stage == "heating" or stage == "desorption" or stage == "steam_desorption":
        P = vacuum_pressure
        dPdz = None
    elif stage == "pressurisation":
        P = None
        dPdz = -tau * (ambient_pressure - pressure_previous_step)
    return P, dPdz


def calculate_gas_heat_capacity(y1, y2, y3, T):
    params_N2 = {
        "A": 0.0000106569,
        "B": -0.0064057779,
        "C": 30.1318,
    }
    params_H2O = {
        "A": 0.00172619108,
        "B": -1.229711840269,
        "C": 250.9868,
    }
    params_CO2 = {
        "A": -0.0000323753,
        "B": 0.0617517420,
        "C": 37.4151,
    }
    params_O2 = {
        "A": 0.00000908259,
        "B": 0.00100528795,
        "C": 28.3,
    }
    Cp_N2 = params_N2["A"] * T**2 + params_N2["B"] * T + params_N2["C"]  # J/(mol·K)
    Cp_H2O = params_H2O["A"] * T**2 + params_H2O["B"] * T + params_H2O["C"]  # J/(mol·K)
    Cp_CO2 = params_CO2["A"] * T**2 + params_CO2["B"] * T + params_CO2["C"]  # J/(mol·K)
    Cp_O2 = params_O2["A"] * T**2 + params_O2["B"] * T + params_O2["C"]  # J/(mol·K)
    Cp_gas = y1 * Cp_CO2 + y2 * Cp_H2O + y3 * Cp_N2 + (1 - y1 - y2 - y3) * Cp_O2
    return Cp_gas  # J/(mol·K)


def calculate_gas_viscosity():
    return 1.8e-5  # Pa·s


def calculate_axial_dispersion_coefficient(bed_props, v_left):
    D_m = bed_props["molecular_diffusivity"]  # m²/s (molecular diffusion)
    v_0 = v_left
    d_p = bed_props["particle_diameter"]
    return 0.7 * D_m + (0.5 * v_0 * d_p)


def calculate_gas_density(P, T):
    R = 8.314
    return P / (R * T)  # mol/m³


def relative_humidity_to_mole_fraction(RH, P, T):
    P_sat = 611.21 * np.exp(
        (18.678 - ((T - 273.15) / 234.5)) * (T - 273.15) / (T - 16.01)
    )  # Tetens equation for water, Pa
    y2 = RH * P_sat / P
    return y2


def mole_fraction_to_relative_humidity(y2, P, T):
    P_sat = 611.21 * np.exp(
        (18.678 - ((T - 273.15) / 234.5)) * (T - 273.15) / (T - 16.01)
    )  # Tetens equation for water, Pa
    RH = y2 * P / P_sat
    return RH


def H2O_boiling_point(P):
    """
    Calculate the boiling point of water at a given pressure using the Antoine equation.

    Args:
        P (float or np.ndarray): Pressure in Pa.

    Returns:
        T_boil (float or np.ndarray): Boiling point temperature in K.
    """
    # Antoine equation parameters for water
    A = 8.07131
    B = 1730.63
    C = 233.426

    # Convert pressure from Pa to mmHg
    P_mmHg = P / 133.322

    # Calculate boiling point in Celsius
    T_boil_C = B / (A - np.log10(P_mmHg)) - C

    # Convert to Kelvin
    T_boil_K = T_boil_C + 273.15

    return T_boil_K


# ============================================================
# 5. BALANCE ERROR FUNCTIONS
# ============================================================


def total_mass_balance_error(F, P, T, n1, n2, time, bed_props, grid):
    """
    Returns % mass balance error for total moles.
    """
    ε = bed_props["bed_voidage"]
    A = bed_props["column_area"]
    R = bed_props["R"]
    z = grid["xCentres"][1:-1]

    mole_in = np.sum(F[:4, -1])
    mole_out = np.sum(F[4:8, -1])

    n_acc_final = ε * A * P[:, -1] / (R * T[:, -1]) + (1 - ε) * A * (
        n1[:, -1] + n2[:, -1]
    )
    n_acc_init = ε * A * P[:, 0] / (R * T[:, 0]) + (1 - ε) * A * (n1[:, 0] + n2[:, 0])

    mole_acc = scipy.integrate.trapezoid(n_acc_final - n_acc_init, z)

    # print("Mole in:", mole_in, "mol")
    # rint("Mole out:", mole_out, "mol")
    # print("Mole accumulated:", mole_acc, "mol")
    return np.abs(mole_in - mole_out - mole_acc) / np.abs(mole_acc) * 100


def CO2_mass_balance_error(F, P, T, y1, n1, time, bed_props, grid):
    """
    Returns % mass balance error for CO₂.
    """
    ε = bed_props["bed_voidage"]
    A = bed_props["column_area"]
    R = bed_props["R"]
    z = grid["xCentres"][1:-1]

    mole_in = F[0, -1]
    mole_out = F[4, -1]

    n_acc_final = (
        ε * A * P[:, -1] * y1[:, -1] / (R * T[:, -1]) + (1 - ε) * A * n1[:, -1]
    )
    n_acc_init = ε * A * P[:, 0] * y1[:, 0] / (R * T[:, 0]) + (1 - ε) * A * n1[:, 0]

    mole_acc = scipy.integrate.trapezoid(n_acc_final - n_acc_init, z)
    return np.abs(mole_in - mole_out - mole_acc) / np.abs(mole_acc) * 100


def energy_balance_error(E, T, P, y1, y2, y3, n1, n2, Tw, time, bed_props, grid):
    """
    Returns % energy balance error.
    """
    ε = bed_props["bed_voidage"]
    A = bed_props["column_area"]
    R = bed_props["R"]
    z = grid["xCentres"][1:-1]
    Cp_g = calculate_gas_heat_capacity()
    Cp_s = bed_props["sorbent_heat_capacity"]

    bed_density = bed_props["bed_density"]  # kg/m³
    Cp_1 = bed_props["heat_capacity_1"]  # J/(mol*K)
    Cp_2 = bed_props["heat_capacity_2"]  # J/(mol*K)

    # Energy terms
    heat_in = E[0, -1]
    heat_out = E[1, -1]

    ΔH1 = cfg.adsorption_isotherm_1(
        P[:, -1], T[:, -1], y1[:, -1], y2[:, -1], y3[:, -1], n1[:, -1], bed_props
    )[1]
    ΔH2 = cfg.adsorption_isotherm_2(P[:, -1], T[:, -1], y2[:, -1], bed_props)[1]

    heat_gen = (
        (1 - ε)
        * A
        * scipy.integrate.simpson(
            (
                np.abs(ΔH1) * (n1[:, -1] - n1[:, 0])
                + np.abs(ΔH2) * (n2[:, -1] - n2[:, 0])
            ),
            x=z,
        )
    )

    heat_acc_solid = (
        A * Cp_s * bed_density * scipy.integrate.simpson((T[:, -1] - T[:, 0]), x=z)
    )
    heat_acc_gas = (
        ε
        * A
        * Cp_g
        * scipy.integrate.simpson(
            (P[:, -1] / (R * T[:, -1]) * T[:, -1] - P[:, 0] / (R * T[:, 0]) * T[:, 0]),
            x=z,
        )
    )
    heat_acc_adsorbed = (
        (1 - ε)
        * A
        * scipy.integrate.trapezoid(
            (
                (Cp_1 * n1[:, -1] + Cp_2 * n2[:, -1]) * T[:, -1]
                - (Cp_1 * n1[:, 0] + Cp_2 * n2[:, 0]) * T[:, 0]
            ),
            z,
        )
    )

    heat_acc_wall = (
        bed_props["wall_density"]
        * bed_props["wall_heat_capacity"]
        * np.pi
        * (bed_props["outer_bed_radius"] ** 2 - bed_props["inner_bed_radius"] ** 2)
        * scipy.integrate.simpson((Tw[:, -1] - Tw[:, 0]), x=z)
    )
    heat_loss_from_bed = E[2, -1]

    # print("Heat in:", heat_in, "J")
    # print("Heat out:", heat_out, "J")
    # print("Heat generation:", heat_gen, "J")
    # print("Heat accumulation in solid:", heat_acc_solid, "J")
    # print("Heat accumulation in gas:", heat_acc_gas, "J")
    # print("Heat accumulation in adsorbed phase:", heat_acc_adsorbed, "J")
    # print("Heat accumulation in wall:", heat_acc_wall, "J")
    # print("Heat loss from bed:", heat_loss_from_bed, "J")
    # print("Delta H1:", ΔH1, "J/mol")
    # print("Bed length", bed_props["bed_length"], "m")
    # print("sum(deltaZ)", np.sum(grid["deltaZ"][1:-1]), "m")

    # print("Bed density", bed_density, "kg/m³")
    # print("Cp_g", Cp_g, "J/(mol*K)")
    # print("Cp_s", Cp_s, "J/(kg*K)")
    # print("Cp_1", Cp_1, "J/(mol*K)")
    # print("Cp_2", Cp_2, "J/(mol*K)")

    total_acc = heat_acc_solid + heat_acc_gas + heat_acc_adsorbed + heat_acc_wall
    return (
        np.abs(heat_in + heat_gen - total_acc - heat_loss_from_bed - heat_out)
        / np.abs(total_acc)
        * 100
    )


def cycle_error(initial_state_vector, final_state_vector):
    cycle_error_vector = final_state_vector - initial_state_vector
    cycle_error = np.dot(cycle_error_vector, np.transpose(cycle_error_vector))

    return cycle_error


# ============================================================
# 6. PLOTTING
# ============================================================


def create_polished_plot(
    time,
    result,
    y_label,
    stage_change_times,
    stage_names,
    save_path=None,
):
    """
    Publication-quality plot of first, middle, and last node of a variable over time.

    Args:
        time (array): Time points.
        result (2D array): Variable values (shape: nodes x time).
        title (str): Plot title.
        y_label (str): Y-axis label.
        save_path (str, optional): If provided, saves the figure to this path.

    result = np.array(result)

    """

    time = np.array(time)
    result = np.array(result)
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.size": 14,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "legend.fontsize": 13,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "lines.linewidth": 2.5,
            "figure.dpi": 150,
            "axes.grid": True,
            "grid.alpha": 0.4,
        }
    )

    plt.figure(figsize=(9, 4))

    # Add vertical lines and stage labels
    stage_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    stage_edges = np.concatenate(([time[0]], stage_change_times))
    for i in range(len(stage_names)):
        mask = (time >= stage_edges[i]) & (time <= stage_edges[i + 1])
        plt.plot(
            time[mask],
            result[mask],
            color=stage_colors[i % len(stage_colors)],
            label=stage_names[i].capitalize(),
        )
        # plt.axvline(stage_edges[i+1], color='grey', linestyle='--', alpha=0.7)

    plt.xlabel("Time (s)", fontsize=16, labelpad=8)
    plt.ylabel(y_label, fontsize=16, labelpad=8)
    plt.tight_layout(pad=2)
    plt.grid(False)
    plt.minorticks_on()
    plt.tick_params(axis="both", which="major", length=6)
    plt.tick_params(axis="both", which="minor", length=3)

    plt.locator_params(axis="y", nbins=3)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


def plot_all_profiles(time, profiles, stage_change_times, stage_names, bed_properties):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.size": 13,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"],
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "lines.linewidth": 1.0,
            "figure.dpi": 150,
        }
    )

    fig, axes = plt.subplots(3, 2, figsize=(9, 7), sharex=True)
    axes = axes.flatten()

    profile_list = [
        ("adsorbed_CO2", "Adsorbed CO$_2$, mol/kg"),
        ("adsorbed_H2O", "Adsorbed H$_2$O, mol/kg"),
        ("temperature", "Fluid temperature, K"),
        ("pressure_outlet", "Pressure, bar"),
        # ("outlet_air", "Air mole fraction"),
        ("RH", "Relative humidity"),
    ]

    # Calculate outlet_air and RH if not present
    outlet_air = np.array(profiles["outlet_N2"]) + np.array(profiles["outlet_O2"])
    RH = mole_fraction_to_relative_humidity(
        np.array(profiles["outlet_H2O"]),
        np.array(profiles["pressure_outlet"]),
        np.array(profiles["temperature"]),
    )
    bed_density = bed_properties["bed_density"]
    bed_voidage = bed_properties["bed_voidage"]

    adsorbed_CO2_kg = np.array(profiles["adsorbed_CO2"]) / (
        bed_density / (1 - bed_voidage)
    )
    adsorbed_H2O_kg = np.array(profiles["adsorbed_H2O"]) / (
        bed_density / (1 - bed_voidage)
    )

    data_dict = dict(profiles)
    data_dict["outlet_air"] = outlet_air
    data_dict["RH"] = RH
    data_dict["adsorbed_CO2"] = adsorbed_CO2_kg
    data_dict["adsorbed_H2O"] = adsorbed_H2O_kg

    stage_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    stage_edges = np.concatenate(([time[0]], stage_change_times))

    outlet_CO2 = np.array(profiles["outlet_CO2"])
    outlet_H2O = np.array(profiles["outlet_H2O"])
    RH = mole_fraction_to_relative_humidity(
        np.array(profiles["outlet_H2O"]),
        np.array(profiles["pressure_outlet"]),
        np.array(profiles["temperature"]),
    )

    data_dict = dict(profiles)
    data_dict["RH"] = RH

    stage_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    stage_edges = np.concatenate(([time[0]], stage_change_times))

    # Plot first four profiles
    for ax, (key, ylabel) in zip(axes[:4], profile_list[:4]):
        for i in range(len(stage_names)):
            mask = (time >= stage_edges[i]) & (time <= stage_edges[i + 1])
            ydata = np.array(data_dict[key])[mask]
            if key == "pressure_outlet":
                ydata = ydata / 1e5
            ax.plot(time[mask], ydata, color=stage_colors[i % len(stage_colors)])
        ax.set_ylabel(ylabel)
        ax.locator_params(axis="y", nbins=5)
        ax.linewidth = 0.1
        ax.grid(False)
        ax.tick_params(axis="both", which="major", length=6)
        ax.tick_params(axis="both", which="minor", length=3)

    # Fifth subplot: outlet CO2 and H2O mole fractions
    ax5 = axes[4]
    for i in range(len(stage_names)):
        mask = (time >= stage_edges[i]) & (time <= stage_edges[i + 1])
        ax5.plot(
            time[mask], outlet_CO2[mask], label="CO$_2$", color="k", linestyle="--"
        )
        ax5.plot(time[mask], outlet_H2O[mask], label="H$_2$O", color="gray")
    ax5.set_ylabel("Gas-phase mole fraction")
    ax5.locator_params(axis="y", nbins=5)
    ax5.grid(False)
    ax5.linewidth = 0.1
    ax5.tick_params(axis="both", which="major", length=6)
    ax5.tick_params(axis="both", which="minor", length=3)
    # ax5.legend(loc="best", frameon=False)

    # Sixth subplot: RH
    ax6 = axes[5]
    for i in range(len(stage_names)):
        mask = (time >= stage_edges[i]) & (time <= stage_edges[i + 1])
        ax6.plot(time[mask], RH[mask], color=stage_colors[i % len(stage_colors)])
    ax6.set_ylabel("-")
    ax6.locator_params(axis="y", nbins=5)
    ax6.grid(False)
    ax6.linewidth = 0.1
    ax6.tick_params(axis="both", which="major", length=6)
    ax6.tick_params(axis="both", which="minor", length=3)

    axes[-1].set_xlabel("Time (s)")
    axes[-2].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.4)  # Increase white space between plots
    plt.show()


def create_quick_plot(time, result, title, y_label):
    """Quick plot of a variable over time."""
    plt.figure(figsize=(6, 4))
    plt.plot(time, result, label="Node", linestyle=None, marker="x", markersize=3)

    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("Cycle number", fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.show()


def create_multi_plot(profiles, bed_properties):
    """
    Create a grid of subplots for multiple time series.

    Parameters
    ----------
    plots : list of tuples
        Each tuple should be (x, y, title, ylabel)
    ncols : int, optional
        Number of columns in the grid (default = 3)
    figsize : tuple, optional
        Size of the whole figure
    """
    # Import data

    time_SB = np.loadtxt(
        "multistage_cycles/stampi-bombelli_profiles/stampi-bombelli_qCO2.csv",
        delimiter=",",
        usecols=0,
    )  # First column
    qCO2_SB = np.loadtxt(
        "multistage_cycles/stampi-bombelli_profiles/stampi-bombelli_qCO2.csv",
        delimiter=",",
        usecols=1,
    )  # Second column

    # time_SB2 = np.loadtxt('multistage_cycles/stampi-bombelli_profiles/stampi-bombelli_yCO2.csv', delimiter=',', usecols=0)  # First column
    # yCO2_SB = np.loadtxt('multistage_cycles/stampi-bombelli_profiles/stampi-bombelli_yCO2.csv', delimiter=',', usecols=1)  # Second column

    time_SB3 = np.loadtxt(
        "multistage_cycles/stampi-bombelli_profiles/stampi-bombelli_qH2O.csv",
        delimiter=",",
        usecols=0,
    )  # First column
    qH2O_SB = np.loadtxt(
        "multistage_cycles/stampi-bombelli_profiles/stampi-bombelli_qH2O.csv",
        delimiter=",",
        usecols=1,
    )  # Second column

    time_SB4 = np.loadtxt(
        "multistage_cycles/stampi-bombelli_profiles/stampi-bombelli_temp.csv",
        delimiter=",",
        usecols=0,
    )  # First column
    temp_SB = (
        np.loadtxt(
            "multistage_cycles/stampi-bombelli_profiles/stampi-bombelli_temp.csv",
            delimiter=",",
            usecols=1,
        )
        + 273.15
    )  # Second column

    time_SB5 = np.loadtxt(
        "multistage_cycles/stampi-bombelli_profiles/stampi-bombelli_RH.csv",
        delimiter=",",
        usecols=0,
    )  # First column
    RH_SB = np.loadtxt(
        "multistage_cycles/stampi-bombelli_profiles/stampi-bombelli_RH.csv",
        delimiter=",",
        usecols=1,
    )  # Second column

    time = profiles["time"]
    T_gas = np.array(profiles["temperature"])
    T_wall = np.array(profiles["wall_temperature"])
    P_inlet = np.array(profiles["pressure_inlet"])
    P_outlet = np.array(profiles["pressure_outlet"])
    outlet_CO2 = np.array(profiles["outlet_CO2"])
    outlet_H2O = np.array(profiles["outlet_H2O"])
    adsorbed_CO2 = np.array(profiles["adsorbed_CO2"])
    adsorbed_H2O = np.array(profiles["adsorbed_H2O"])
    # outlet_N2 = np.array(profiles["outlet_N2"])
    # outlet_O2 = np.array(profiles["outlet_O2"])
    # equilibrium_CO2 = np.array(profiles["equilibrium_CO2"])

    bed_density = bed_properties["bed_density"]
    bed_voidage = bed_properties["bed_voidage"]

    adsorbed_CO2 = np.array(adsorbed_CO2) / (bed_density / (1 - bed_voidage))
    adsorbed_H2O = np.array(adsorbed_H2O) / (bed_density / (1 - bed_voidage))
    relative_humidity = mole_fraction_to_relative_humidity(outlet_H2O, P_outlet, T_gas)

    figsize = (15, 8)
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.ravel()

    # 1. Temperature overlay (gas and wall)
    ax = axes[0]
    ax.plot(time, T_gas, label="Gas Temp (mid)", color="tab:blue")
    ax.plot(time, T_wall, label="Wall Temp (mid)", color="tab:orange")
    ax.plot(
        time_SB4,
        temp_SB,
        label="WADST (Young et al.)",
        color="black",
        linestyle="--",
        alpha=0.7,
    )
    ax.set_title("Temperature Profiles")
    ax.set_ylabel("Temperature (K)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Pressure overlay (inlet and outlet)
    ax = axes[1]
    ax.plot(time, P_inlet, label="Inlet Pressure", color="tab:green")
    ax.plot(time, P_outlet, label="Outlet Pressure", color="tab:red")
    ax.set_title("Pressure Profiles")
    ax.set_ylabel("Pressure (Pa)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. CO2 gas phase (outlet and mid)
    ax = axes[2]
    ax.plot(time, outlet_CO2, label="CO2 Outlet", color="tab:purple")
    # ax.plot(time_SB2, yCO2_SB, label="WADST (Young et al.)", color="black", linestyle="--", alpha=0.7)
    ax.set_title("CO2 Gas Phase")
    ax.set_ylabel("CO2 Mole Fraction")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. CO2 adsorbed
    ax = axes[3]
    ax.plot(time, adsorbed_CO2, label="CO2 Adsorbed (mid)", color="tab:blue")
    # ax.plot(
    #     time_SB,
    #     qCO2_SB,
    #     label="Stampi-Bombelli",
    #     color="black",
    #     linestyle="--",
    #     alpha=0.7,
    # )
    # ax.plot(time, equilibrium_CO2, label="CO2 Equilibrium (mid)", color="tab:orange", linestyle="--")
    ax.set_title("CO2 Adsorbed")
    ax.set_ylabel("CO2 Loading (mol/kg)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Relative Humidity
    ax = axes[4]
    # x.plot(time, outlet_H2O, label="H2O Outlet", color="tab:purple")
    # ax.plot(time, outlet_N2, label="N2 Outlet", color="tab:green")
    # ax.plot(time, outlet_O2, label="O2 Outlet", color="tab:orange")
    ax.plot(time, relative_humidity, label="Relative Humidity", color="tab:purple")
    # ax.plot(
    #     time_SB5,
    #     RH_SB,
    #     label="WADST (Young et al.)",
    #     color="black",
    #     linestyle="--",
    #     alpha=0.7,
    # )
    ax.set_title("Relative Humidity")
    ax.set_ylabel("Relative Humidity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. H2O adsorbed
    ax = axes[5]
    ax.plot(time, adsorbed_H2O, label="H2O Adsorbed (mid)", color="tab:blue")
    # ax.plot(
    #     time_SB3,
    #     qH2O_SB,
    #     label="WADST (Young et al.)",
    #     color="black",
    #     linestyle="--",
    #     alpha=0.7,
    # )
    ax.set_title("H2O Adsorbed")
    ax.set_ylabel("H2O Loading (mol/kg)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Remove gridlines
    for ax in axes:
        ax.grid(False)

        # Add dashed vertical lines for stage transitions
    # Define stage durations (must match your run_cycle stages)
    stage_durations = [
        bed_properties["adsorption_time"],
        bed_properties["blowdown_time"],
        bed_properties["heating_time"],
        bed_properties["desorption_time"],
        bed_properties["pressurisation_time"],
        # Add cooling time if used
    ]
    stage_start_times = [0]
    for dur in stage_durations[:-1]:
        stage_start_times.append(stage_start_times[-1] + dur)

    for ax in axes:
        for t in stage_start_times:
            ax.axvline(x=t, color="#cccccc", linestyle="--", alpha=0.2)

    for ax in axes:
        ax.set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show()


# ============================================================
# 7. TESTING FUNCTIONS
# ============================================================


def product_mass(F, t_cycle, bed_properties):
    mass_CO2_product = F[4, -1] * bed_properties["MW_1"] / 1000  # kg

    mass_carrier_gas_product = (
        F[5, -1] * bed_properties["MW_2"] / 1000
        + F[6, -1] * bed_properties["MW_3"] / 1000
        + F[7, -1] * bed_properties["MW_4"] / 1000
    )  # kg

    mass_CO2_in = F[0, -1] * bed_properties["MW_1"] / 1000  # kg

    mass_H2O_out = F[5, -1] * bed_properties["MW_2"] / 1000  # kg

    return mass_CO2_product, mass_carrier_gas_product, mass_CO2_in, mass_H2O_out


def product_mols(F, t_cycle, bed_properties):
    mol_CO2_product = F[4, -1]  # mol

    mol_carrier_gas_product = F[5, -1] + F[6, -1] + F[7, -1]  # mol

    mol_CO2_in = F[0, -1]

    mol_H2O_out = F[5, -1]

    total_mols = np.sum(F[:, -1])

    return mol_CO2_product, mol_carrier_gas_product, mol_CO2_in, mol_H2O_out, total_mols
