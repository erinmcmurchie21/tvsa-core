import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

# ============================================================
# 1. GRID DEFINITION
# ============================================================

def create_non_uniform_grid(bed_properties):
    """
    Create a 1D non-uniform grid with ghost cells for a column model.
    
    Returns:
        column_grid (dict): grid properties including xCentres, xWalls, deltaZ, etc.
    """
    nx = 30          # Number of physical cells
    nGhost = 1       # Number of ghost cells on each end
    xmin, xmax = 0, bed_properties["bed_length"]  # Column length from bed properties

    dX_wide = 2 * (xmax - xmin) / (2 * nx - 9)
    dX_small = dX_wide / 4

    xGhost_start = xmin - dX_small * (np.flip(np.arange(nGhost)) + 0.5)
    xFirst = xmin + (np.arange(3) + 0.5) * dX_small
    xCentral = xmin + xFirst[-1] + (np.arange(nx - 6) + 1/8 + 0.5) * dX_wide
    xEnd = xmin + xCentral[-1] + (1/8 + 0.5) * dX_wide + dX_small * np.arange(3)
    xGhost_end = xmax + dX_small * (np.arange(nGhost) + 0.5)

    xCentres = np.concatenate((xGhost_start, xFirst, xCentral, xEnd, xGhost_end))

    xWalls_s = np.arange(-nGhost, 4) * dX_small
    xWalls_m = xWalls_s[-1] + np.arange(1, nx - 5) * dX_wide
    xWalls_e = xWalls_m[-1] + np.arange(1, 3 + nGhost + 1) * dX_small
    xWalls = np.concatenate((xWalls_s, xWalls_m, xWalls_e))
    deltaZ = xWalls[1:nx + 2 * nGhost + 1] - xWalls[:nx + 2 * nGhost]

    return {
        "num_cells": nx,
        "nGhost": nGhost,
        "xCentres": xCentres,
        "xWalls": xWalls,
        "deltaZ": deltaZ
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
    return (y1 * (x0 - x2)**2 - y0 * (x1 - x2)**2) / ((x0 - x1) * (x0 + x1 - 2 * x2))

def quadratic_extrapolation_derivative_nonzero(x0, y0, x1, y1, x2, a, b):
    """
    Extrapolate value y2 such that dy/dx = a * (b - y2) at x2.
    Used for inlet/outlet Dirichlet or Robin BCs.
    """
    return (a * b + (y0 - y1) / (x0 - x1) + y0 / (-x0 + x2) + y1 / (-x1 + x2)) / (a + 1 / (-x0 + x2) + 1 / (-x1 + x2))

# ============================================================
# 3. ISOTHERM EQUILIBRIA
# ============================================================

def adsorption_isotherm_1(pressure, temperature, y1, y2, y3, n1, bed_properties, isotherm_type="Toth"):
    """
    Toth isotherm for CO₂ on solid sorbent.

    Returns:
        equilibrium_loading (mol/m³ of solid)
        adsorption_heat_1 (J/mol)
    """
    R = 8.314           # J/(mol·K)
    bed_density = bed_properties["bed_density"]          # kg/m³
    ε = bed_properties["bed_voidage"]           # bed void fraction

    if isotherm_type == "Toth":
        T_0 = 296           # Reference temperature (K)
        n_s = 2.38 * np.exp(0 * (1 - temperature / T_0))         # mol/kg
        b = 70.74 * np.exp(-1*(-57047) / (R * T_0) * (1-T_0 / temperature))  # kPa⁻¹
        t = 0.4148 - 1.606 * (1 - T_0 / temperature)
        pressure_kPa = pressure / 1000  # Convert pressure from Pa to kPa

        load_kg = n_s * b * pressure_kPa * y1 / (1 + (b * pressure_kPa * y1)**t)**(1 / t)  # mol/kg
        load_m3 = load_kg * bed_density / (1 - ε)  # mol/m³
        ΔH = -57047  # J/mol
    
    elif isotherm_type == "Langmuir":
        c1 = y1 * pressure / (R * temperature)  # mol/m3
        c2 = y3 * pressure / (R * temperature)  # mol/m3
        b1 = 8.65e-7 * np.exp(-(-36641.21) / (R * temperature)) #m3/mol
        b2 = 2.5e-6 * np.exp(-(-1.58e4) / (R * temperature)) #m3/mol
        d1 = 2.63e-8 * np.exp(-(-3590.66)/ (R * temperature)) #m3/mol
        d2 = 0
        
        load_kg = (3.09 * b1 * c1 / (1 + b1 * c1 + b2 * c2) + 
                                     2.54 * d1 * c1 / (1 + d1 * c1 + d2 * c2))  # mol/kg
        load_m3 = load_kg * bed_density / (1 - ε)  # mol/m³
        ΔH = -57047  # J/mol

    elif isotherm_type == "Dual_site Langmuir":
        c1 = y1 * pressure / (R * temperature)  # mol/m3
        b1 = 3.17e-6 * np.exp(-(-28.63e3) / (R * temperature)) #m3/mol
        b2 = 3.21e-6 * np.exp(-(-20.37e3) / (R * temperature)) #m3/mol
        qs1 = 0.44 #mol/kg
        qs2 = 6.10

        load_kg = (qs1 * b1 * c1 / (1 + b1 * c1) + 
                                     qs2 * b2 * c1 / (1 + b2 * c1))  # mol/kg

        load_m3 = load_kg * bed_density / (1 - ε)  # mol/m³
        ΔH = R * (-3391.58 + 273.2725 * n1 * (1 - bed_properties["bed_voidage"])/bed_density) # J/mol

    return load_m3, ΔH

def adsorption_isotherm_2(pressure, temperature, y2, bed_properties, isotherm_type="GAB"):
    """
    GAB isotherm for H₂O.

    Returns:
        equilibrium_loading (mol/m³ of solid)
        adsorption_heat_2 (J/mol)
    """
    R = 8.314 # J/mol·K
    bed_density = bed_properties["bed_density"]  # kg/m³
    ε = bed_properties["bed_voidage"]  # bed void fraction

    if isotherm_type == "GAB":
        K_ads = 0.5751 # -
        c_m = 36.48 # mol/kg
        c_G = 0.1489 # -

        P_sat = 10**(8.07131-1730.63/(233.426+(temperature-273.15)))  # Pressure in mmHg, Antoine equation for water, Ward et al. 2024
        P_sat_Pa = P_sat * 133.322  # Convert mmHg to Pa
        RH = y2 * pressure / (P_sat_Pa)  # dimensionless

        load_kg = c_m * c_G * K_ads * RH / ((1 - K_ads * RH) * (1 + (c_G - 1) * K_ads * RH))  # mol/kg
        load_m3 = load_kg * bed_density / (1 - ε)  # mol/m³

    elif isotherm_type == "None":
        load_m3 = 0 * pressure

    ΔH = -49000  # J/mol
    return load_m3, ΔH

# ============================================================
# 4. PHYSICAL PROPERTY MODELS
# ============================================================

def calculate_gas_heat_capacity():
    return 840 * 44.01 / 1000  # J/mol·K (approx. for air/CO2)

def heat_transfer_coefficient():
    return 140, 20  # W/m²·K (bed, wall)

def calculate_gas_viscosity():
    return 1.8e-5  # Pa·s

def calculate_gas_thermal_conductivity():
    return 1  # W/(m·K)

def calculate_wall_thermal_conductivity():
    return 205  # W/(m·K)

def calculate_axial_dispersion_coefficient(bed_props, inlet_vals):
    D_m = bed_props["molecular_diffusivity"]  # m²/s (molecular diffusion)
    v_0 = inlet_vals["velocity"]
    d_p = bed_props["particle_diameter"]
    return 0.7 * D_m + (0.5 * v_0 * d_p)

def calculate_gas_density(P, T):
    R = 8.314
    return P / (R * T)  # mol/m³

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

    n_acc_final = ε * A * P[:, -1] / (R * T[:, -1]) + (1 - ε) * A * (n1[:, -1] + n2[:, -1])
    n_acc_init = ε * A * P[:, 0] / (R * T[:, 0]) + (1 - ε) * A * (n1[:, 0] + n2[:, 0])

    mole_acc = scipy.integrate.trapezoid(n_acc_final - n_acc_init, z)
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
    mole_out = F[4,-1]

    n_acc_final = ε * A * P[:, -1] * y1[:, -1] / (R * T[:, -1]) + (1 - ε) * A * n1[:, -1]
    n_acc_init = ε * A * P[:, 0] * y1[:, 0] / (R * T[:, 0]) + (1 - ε) * A * n1[:, 0]

    mole_acc = scipy.integrate.trapezoid(n_acc_final - n_acc_init, z)
    return np.abs(mole_in - mole_out - mole_acc) / np.abs(mole_acc) * 100

def energy_balance_error(E, T, P, y1, y2, y3, n1, n2, time, bed_props, grid):
    """
    Returns % energy balance error.
    """
    ε = bed_props["bed_voidage"]
    A = bed_props["column_area"]
    R = bed_props["R"]
    z = grid["xCentres"][1:-1]
    Cp_g = calculate_gas_heat_capacity()
    Cp_s = bed_props["sorbent_heat_capacity"]
    bed_density = bed_props["bed_density"] # kg/m³
    Cp_1 = bed_props["heat_capacity_1"] # J/(mol*K)
    Cp_2 = bed_props["heat_capacity_2"] # J/(mol*K)

    # Energy terms
    heat_in = E[0, -1]
    heat_out = E[1, -1]

    ΔH1 = adsorption_isotherm_1(P[:, -1], T[:, -1], y1[:, -1], y2[:, -1], y3[:,-1], n1[:,-1], bed_props)[1]
    ΔH2 = adsorption_isotherm_2(P[:, -1], T[:, -1], y2[:, -1], bed_props)[1]

    heat_gen = (1 - ε) * A * scipy.integrate.trapezoid(
        np.abs(ΔH1) * (n1[:, -1] - n1[:, 0]) +
        np.abs(ΔH2) * (n2[:, -1] - n2[:, 0]), z)

    heat_acc_solid = (1 - ε) * A * Cp_s * bed_density * scipy.integrate.trapezoid((T[:, -1] - T[:, 0]), z)
    heat_acc_gas = ε * A * Cp_g * scipy.integrate.trapezoid(
        P[:, -1]/(R*T[:, -1]) * T[:, -1] - P[:, 0]/(R*T[:, 0]) * T[:, 0], z)
    heat_acc_adsorbed = (1 - ε) * A * scipy.integrate.trapezoid(
        (Cp_1*n1[:, -1] + Cp_2*n2[:, -1]) * T[:, -1] - (Cp_1*n1[:, 0] + Cp_2*n2[:, 0]) * T[:, 0], z)
    
    heat_loss_from_bed = 0 # E[2,-1]

    total_out = heat_out + heat_acc_solid + heat_acc_gas + heat_acc_adsorbed + heat_loss_from_bed
    return np.abs(heat_in + heat_gen - total_out) / np.abs(total_out) * 100

# ============================================================
# 6. PLOTTING
# ============================================================

def create_plot(time, result, title, y_label):
    """Quick plot of first, middle, and last node of a variable over time."""
    plt.figure(figsize=(6, 4))
    for idx, label in zip([0, 14, 29], ['First node', 'Central node', 'Final node']):
        plt.plot(time, result[idx], label=label, linewidth=2, marker='o', markersize=3)

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.show()

# Create a combined plot with all 6 variables
def create_combined_plot(time, T_result, P_result, y1_result, n1_result, y1_walls_result, v_walls_result, bed_properties):
    """Create a combined plot with all 6 variables in subplots."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    fig.suptitle('Adsorption Column Simulation Results', fontsize=20, fontweight='bold')
    
    # Define the data and labels for each subplot
    plot_data = [
        
        (y1_result, "Gas phase CO2 against time", "Gas Phase CO2 (mol fraction)", axes[0, 2]),
        (T_result, "Temperature against time", "Temperature (K)", axes[0, 0]),
        (n1_result, "CO2 loading against time", "Loading CO2 (mol/m³)", axes[1, 0]),
        (P_result, "Pressure against time", "Pressure (Pa)", axes[0, 1]),
        

        (y1_walls_result, "Gas phase CO2 at exit against time", "Gas Phase CO2 (mol fraction)", axes[1, 1]),
        (v_walls_result * 60 * 1e6 * bed_properties["column_area"] * bed_properties["bed_voidage"], "Outlet flow rate", "Exit flowrate (cm³/s)", axes[1, 2])
    ]
    
    # Plot each variable
    for result, title, ylabel, ax in plot_data[0:4]:
        for idx, label in zip([0, 9, 29], ['First node', 'Central node', 'Final node']):
            ax.plot(time, result[idx], label=label, linewidth=2, marker='o', markersize=3)

    for result, title, ylabel, ax in plot_data[4:]:
        ax.plot(time, result[-1], linewidth=2, marker='o', markersize=3)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()