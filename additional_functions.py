import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt


def create_non_uniform_grid():
    nx = 30
    nGhost = 1
    xmin = 0
    xmax = 1.0

    dX_wide = 2 * (xmax - xmin) / (2 * nx - 9)
    dX_small = dX_wide / 4
    xGhost_start = xmin - dX_small * (np.flip(np.arange(nGhost)) + 0.5)
    xFirst = xmin + (np.arange(3) + 0.5) * dX_small
    xCentral = xmin + xFirst[-1] + (np.arange(nx - 6) + 1 / 8 + 1 / 2) * dX_wide
    xEnd = xmin + xCentral[-1] + (1 / 8 + 1 / 2) * dX_wide + dX_small * (np.arange(3))
    xGhost_end = xmax + dX_small * (np.arange(nGhost) + 0.5)

    xCentres = np.concatenate((xGhost_start, xFirst, xCentral, xEnd, xGhost_end))

    xWalls_s = np.arange(-nGhost, 4) * dX_small
    xWalls_m = xWalls_s[-1] + np.arange(1, nx - 5) * dX_wide
    xWalls_e = xWalls_m[-1] + np.arange(1, 3 + nGhost + 1) * dX_small
    xWalls = np.concatenate((xWalls_s, xWalls_m, xWalls_e))
    deltaZ = xWalls[1 : nx + 2 * nGhost + 1] - xWalls[: nx + 2 * nGhost]

    column_grid = {
        "num_cells": nx,
        "nGhost": nGhost,
        "xCentres": xCentres,
        "xWalls": xWalls,
        "deltaZ": deltaZ,
    }
    return column_grid


def quadratic_extrapolation(x0, y0, x1, y1, x2, y2, x):
    """Quadratic extrapolation for ghost cells"""
    # Calculate the coefficients of the quadratic polynomial
    # Y = y0*L_0(x) + y1*L_1(x) + y2*L_2(x)
    L_0 = (x - x1) * (x - x2) / ((x0 - x1) * (x0 - x2))
    L_1 = (x - x0) * (x - x2) / ((x1 - x0) * (x1 - x2))
    L_2 = (x - x0) * (x - x1) / ((x2 - x0) * (x2 - x1))
    Y = y0 * L_0 + y1 * L_1 + y2 * L_2
    return Y
    # Calculate the value of the polynomial at the ghost cell location


def quadratic_extrapolation_derivative(x0, y0, x1, y1, x2):
    # We want to solve for value y2, given that dy/dx = 0 at x2
    y2 = (y1 * (x0 - x2) ** 2 - y0 * (x1 - x2) ** 2) / ((x0 - x1) * (x0 + x1 - 2 * x2))
    return y2


def quadratic_extrapolation_derivative_nonzero(x0, y0, x1, y1, x2, a, b):
    # We want to solve for value y2, given that dy/dx = 0 at x2
    # d/dx = a * (b - y2)
    y2 = (a * b + (y0 - y1) / (x0 - x1) + y0 / (-x0 + x2) + y1 / (-x1 + x2)) / (
        a + 1 / (-x0 + x2) + 1 / (-x1 + x2)
    )
    return y2


def adsorption_isotherm_1(pressure, temperature, mole_fraction_1, mole_fraction_2):
    # adsorption isotherm for CO2
    R = 8.314  # J/(mol*K)
    sorbent_density = 55.4  # kg/m3
    bed_voidage = 0.4  # Example value for bed voidage

    T_0 = 296  # K Reference temperature
    n_s = 2.38 * np.exp(0 * (1 - temperature / T_0))  # mol/kg
    b = (
        70.74 / 1000 * np.exp(-57.047 * 1000 / (8.314 * T_0) * (T_0 / temperature - 1))
    )  # kPa^-1
    t = 0.4148 + -1.606 * (1 - T_0 / temperature)

    equilibrium_loading_per_kg = (
        n_s
        * b
        * pressure
        * mole_fraction_1
        / (1 + (b * pressure * mole_fraction_1) ** t) ** (1 / t)
    )  # mol/kg
    equilibrium_loading = (
        equilibrium_loading_per_kg * sorbent_density / (1 - bed_voidage)
    )  # Convert to mol/m³

    adsorption_heat_1 = -57 * 1000  # J/mol
    return equilibrium_loading, adsorption_heat_1


def adsorption_isotherm_2(pressure, temperature, mole_fraction_2, isotherm_type="GAB"):
    R = 8.314  # J/(mol*K)
    sorbent_density = 55.4  # kg/m3
    bed_voidage = 0.4  # Example value for bed voidage

    if isotherm_type == "GAB":
        K_ads = 0.5751
        c_m = 36.48  # mol/kg
        c_G = 0.1489
        saturation_pressure = 6.1094 * np.exp(
            17.625 * (temperature - 273.15) / ((temperature - 273.15) + 243.04)
        )  # hPa
        RH = (
            mole_fraction_2 * pressure / (saturation_pressure * 100)
        )  # Relative humidity

        equilibrium_loading_per_kg = (
            c_m * c_G * K_ads * RH / ((1 - K_ads * RH) * (1 + (c_G - 1) * K_ads * RH))
        )  # mol/kg
        # mol/kg * kg/m3 --> mol/m3
        equilibrium_loading = (
            equilibrium_loading_per_kg * sorbent_density / (1 - bed_voidage)
        )  # Convert to mol/m³

    elif isotherm_type == "None":
        equilibrium_loading = 0  # mol/m³

    adsorption_heat_2 = -49 * 1000  # J/mol
    return equilibrium_loading, adsorption_heat_2


def calculate_gas_heat_capacity():
    # Example function to calculate gas heat capacity
    # This is a placeholder and should be replaced with actual calculations or data
    Cp_g = 29.19  # J/(mol*K) for air at room temperature
    return Cp_g


def heat_transfer_coefficient():
    heat_transfer_coefficient = 3
    wall_heat_transfer_coefficient = (
        26  # W/(m^2*K) Example value for wall heat transfer coefficient
    )
    return heat_transfer_coefficient, wall_heat_transfer_coefficient


def calculate_gas_viscosity():
    mu = 1.8e-5
    return mu


def calculate_gas_thermal_conductivity():
    # Example function to calculate gas thermal conductivity
    # This is a placeholder and should be replaced with actual calculations or data
    K_z = 0.09  # W/(m*K*s) for air at room temperature
    return K_z


def calculate_gas_mass_transfer():
    # Example function to calculate gas mass transfer coefficient
    # This is a placeholder and should be replaced with actual calculations or data
    mass_transfer_1 = 0.0002  # s^-1 for CO2
    mass_transfer_2 = 0.002  # s^-1 for H2O
    return mass_transfer_1, mass_transfer_2


def calculate_wall_thermal_conductivity():
    # Example function to calculate wall thermal conductivity
    # This is a placeholder and should be replaced with actual calculations or data
    wall_conductivity = 16  # W/(m*K*s) for the wall material
    return wall_conductivity


def calculate_axial_dispersion_coefficient(bed_properties, inlet_values):
    # Example function to calculate axial dispersion coefficient
    # This is a placeholder and should be replaced with actual calculations or data
    D_m = 1.60e-5  # m^2/s for molecular diffusion coefficient
    v_0 = inlet_values["velocity"]
    d_p = bed_properties["particle_diameter"]  # m, particle diameter
    D_l = 0.7 * D_m * (0.5 * v_0 * d_p)
    return D_l


def calculate_gas_density(P, T):
    # Example function to calculate gas density
    # This is a placeholder and should be replaced with actual calculations or data
    R = 8.314  # Universal gas constant in J/(mol*K)
    rho_gas = P / (R * T)  # mol/m^3 for air at room temperature and pressure

    # CHECK WHETHER THIS SHOULD BE IN MOL/M3

    return rho_gas


def total_mass_balance_error(
    F_result,
    P_result,
    T_result,
    n1_result,
    n2_result,
    time,
    bed_properties,
    column_grid,
):
    bed_voidage = bed_properties["bed_voidage"]
    column_area = bed_properties["column_area"]
    R = bed_properties["R"]
    z = column_grid["xCentres"][1:-1]  # Use correct key name

    # F_result should have shape (8, n_time_points)
    # Integrate over time for inlet and outlet flows
    mole_in = scipy.integrate.trapezoid(
        F_result[0, :] + F_result[1, :] + F_result[2, :] + F_result[3, :], time
    )
    mole_out = scipy.integrate.trapezoid(
        F_result[4, :] + F_result[5, :] + F_result[6, :] + F_result[7, :], time
    )

    # Calculate mole accumulation (difference between final and initial states)
    # Integrate over space (z direction) for final and initial states
    final_moles = scipy.integrate.trapezoid(
        (
            bed_voidage * column_area * P_result[:, -1] / (R * T_result[:, -1])
            + (1 - bed_voidage) * column_area * (n1_result[:, -1] + n2_result[:, -1]),
            z,
        )
    )

    initial_moles = scipy.integrate.trapezoid(
        (
            bed_voidage * column_area * P_result[:, 0] / (R * T_result[:, 0])
            + (1 - bed_voidage) * column_area * (n1_result[:, 0] + n2_result[:, 0])
        ),
        z,
    )

    mole_acc = final_moles - initial_moles

    # Calculate mass balance error as percentage
    mass_balance_error = np.abs(mole_in - mole_out - mole_acc) / np.abs(mole_acc) * 100

    return mass_balance_error


def CO2_mass_balance_error(
    F_result,
    P_result,
    T_result,
    y1_result,
    n1_result,
    time,
    bed_properties,
    column_grid,
):
    bed_voidage = bed_properties["bed_voidage"]
    column_area = bed_properties["column_area"]
    R = bed_properties["R"]
    z = column_grid["xCentres"][1:-1]  # Use correct key name

    # F_result should have shape (8, n_time_points)
    # Integrate over time for inlet and outlet flows
    mole_in = scipy.integrate.trapezoid(F_result[0, :], time)
    mole_out = scipy.integrate.trapezoid(
        F_result[4, :], time
    )  # Assuming F[1] is outlet flow

    # Calculate mole accumulation (difference between final and initial states)
    # Integrate over space (z direction) for final and initial states
    final_moles = scipy.integrate.trapezoid(
        (
            bed_voidage
            * column_area
            * P_result[:, -1]
            * y1_result[:, -1]
            / (R * T_result[:, -1])
            + (1 - bed_voidage) * column_area * n1_result[:, -1]
        ),
        z,
    )

    initial_moles = scipy.integrate.trapezoid(
        (
            bed_voidage
            * column_area
            * P_result[:, 0]
            * y1_result[:, 0]
            / (R * T_result[:, 0])
            + (1 - bed_voidage) * column_area * n1_result[:, 0]
        ),
        z,
    )

    mole_acc = final_moles - initial_moles

    # Calculate mass balance error as percentage
    mass_balance_error = np.abs(mole_in - mole_out - mole_acc) / np.abs(mole_acc) * 100

    return mass_balance_error


def energy_balance_error(
    E_result,
    T_result,
    P_result,
    y1_result,
    y2_result,
    n1_result,
    n2_result,
    time,
    bed_properties,
    column_grid,
):
    bed_voidage = bed_properties["bed_voidage"]
    column_area = bed_properties["column_area"]
    R = bed_properties["R"]
    z = column_grid["xCentres"][1:-1]  # Use correct key name
    Cp_g = (
        calculate_gas_heat_capacity()
    )  # Placeholder gas heat capacity J/(mol*K) - replace with actual calculation
    Cp_solid = bed_properties["sorbent_heat_capacity"]  # J/(kg*K)
    Cp_ads = bed_properties["sorbent_heat_capacity"]  # J/(kg*K)

    # Heat flows integrated over time
    heat_in = scipy.integrate.trapezoid(E_result[0, :], time)
    heat_out = scipy.integrate.trapezoid(
        E_result[1, :], time
    )  # Assuming E[1] is heat out

    # Heat generation from adsorption (integrated over space)
    deltaH_1 = adsorption_isotherm_1(
        P_result[:, -1], T_result[:, -1], y1_result[:, -1], y2_result[:, -1]
    )[1]  # Convert kJ/mol to J/mol
    deltaH_2 = adsorption_isotherm_2(
        P_result[:, -1], T_result[:, -1], y2_result[:, -1]
    )[1]  # Convert kJ/mol to J/mol

    heat_gen = (
        (1 - bed_voidage)
        * column_area
        * scipy.integrate.trapezoid(
            (
                np.abs(deltaH_1) * (n1_result[:, -1] - n1_result[:, 0])
                + np.abs(deltaH_2) * (n2_result[:, -1] - n2_result[:, 0])
            ),
            z,
        )
    )

    # Heat accumulation terms (integrated over space)
    heat_acc_solid = (
        (1 - bed_voidage)
        * column_area
        * Cp_solid
        * scipy.integrate.trapezoid((T_result[:, -1] - T_result[:, 0]), z)
    )

    heat_acc_gas = (
        bed_voidage
        * column_area
        * Cp_g
        * scipy.integrate.trapezoid(
            (
                P_result[:, -1] / (R * T_result[:, -1]) * T_result[:, -1]
                - P_result[:, 0] / (R * T_result[:, 0]) * T_result[:, 0]
            ),
            z,
        )
    )

    heat_acc_adsorbed = (
        (1 - bed_voidage)
        * column_area
        * Cp_ads
        * scipy.integrate.trapezoid(
            (
                (n1_result[:, -1] + n2_result[:, -1]) * T_result[:, -1]
                - (n1_result[:, 0] + n2_result[:, 0]) * T_result[:, 0]
            ),
            z,
        )
    )

    total_heat_out = heat_out + heat_acc_solid + heat_acc_gas + heat_acc_adsorbed

    if np.abs(total_heat_out) > 1e-12:  # Avoid division by zero
        energy_balance_error = (
            np.abs(heat_in + heat_gen - total_heat_out) / np.abs(total_heat_out) * 100
        )
    else:
        energy_balance_error = np.abs(heat_in + heat_gen - total_heat_out) * 100

    return energy_balance_error


def create_plot(time, result, title, y_label):
    # Create the plot for temperature against time
    plt.figure(figsize=(6, 4))
    data1 = result[0]
    data2 = result[14]
    data3 = result[29]

    # Plot each dataset
    plt.plot(time, data1, label="First node", linewidth=2, marker="o", markersize=3)
    plt.plot(time, data2, label="Central node", linewidth=2, marker="s", markersize=3)
    plt.plot(time, data3, label="Final node", linewidth=2, marker="^", markersize=3)

    # Customize the plot
    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("Time", fontsize=12)

    plt.ylabel(y_label, fontsize=12)
    plt.legend(fontsize=11)

    plt.grid(True, alpha=0.3)

    # Show the plot
    plt.show()
