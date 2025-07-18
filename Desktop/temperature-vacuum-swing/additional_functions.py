import numpy as np


def create_non_uniform_grid():
    nx = 30
    nGhost = 1
    xmin = 0
    xmax = 1.0


    dX_wide = 2 * (xmax-xmin) / (2*nx - 9)
    dX_small = dX_wide / 4
    xGhost_start = xmin - dX_small * (np.flip(np.arange(nGhost))+0.5)
    xFirst = xmin + (np.arange(3)+0.5)*dX_small
    xCentral = xmin + xFirst[-1] + (np.arange(nx-6)+1/8+1/2)*dX_wide
    xEnd = xmin + xCentral[-1] + (1/8+1/2)*dX_wide + dX_small*(np.arange(3))
    xGhost_end = xmax + dX_small * (np.arange(nGhost)+0.5)

    xCentres = np.concatenate((xGhost_start, xFirst, xCentral, xEnd, xGhost_end))

    xWalls_s = np.arange(-nGhost,4)*dX_small
    xWalls_m = xWalls_s[-1] + np.arange(1,nx-5)*dX_wide
    xWalls_e = xWalls_m[-1] + np.arange(1,3+nGhost+1)*dX_small
    xWalls = np.concatenate((xWalls_s, xWalls_m, xWalls_e))
    deltaZ = xWalls[1:nx+2*nGhost+1] - xWalls[:nx+2*nGhost]

    column_grid = {
        "num_cells": nx,
        "nGhost": nGhost,
        "xCentres": xCentres,
        "xWalls": xWalls,
        "deltaZ": deltaZ
    }
    return column_grid

def quadratic_extrapolation(x0, y0, x1, y1, x2, y2, x):
    """Quadratic extrapolation for ghost cells"""
    # Calculate the coefficients of the quadratic polynomial
    #Y = y0*L_0(x) + y1*L_1(x) + y2*L_2(x)
    L_0 = (x-x1)*(x-x2)/((x0-x1)*(x0-x2))
    L_1 = (x-x0)*(x-x2)/((x1-x0)*(x1-x2))
    L_2 = (x-x0)*(x-x1)/((x2-x0)*(x2-x1))
    Y = y0*L_0 + y1*L_1 + y2*L_2
    return Y
    # Calculate the value of the polynomial at the ghost cell location

def quadratic_extrapolation_derivative(x0, y0, x1, y1, x2):
    # We want to solve for value y2, given that dy/dx = 0 at x2
    y2 = (y1*(x0-x2)**2-y0*(x1-x2)**2)/((x0-x1)*(x0+x1-2*x2))
    return y2

def adsorption_isotherm_1(pressure, temperature, mole_fraction_1, mole_fraction_2):
    # adsorption isotherm for CO2
    T_0 = 296 #K Reference temperature
    n_s = 2.38 * np.exp(0 * (1-temperature/T_0)) # adsorption equilibrium constant mol/kg
    b = 70.74 / 1000 * np.exp(-57.047 * 1000 / (8.314 * T_0) * (T_0 / temperature -1) ) #kPa^-1
    t = 0.4148 + -1.606*(1-T_0/temperature)
    equilibrium_loading = n_s*b*pressure*mole_fraction_1 / (1+(b*pressure*mole_fraction_1)**t)**(1/t)
    adsorption_heat_1 = -57
    return equilibrium_loading, adsorption_heat_1

def adsorption_isotherm_2(pressure, temperature, mole_fraction):
    #adsorption isotherm for H2O
    K_ads = 0.5751 # adsorption equilibrium constant
    c_m = 36.48 #mol/kg
    c_G = 0.1489
    saturation_pressure = 6.1094 * np.exp(17.625 * (temperature - 273.15) / ((temperature-273.15) + 243.04))  # hPa (100 Pa), saturation pressure of water vapor
    RH = mole_fraction*pressure/(saturation_pressure*100) # Relative humidity
    R = 8.314 # J/(mol*K)
    equilibrium_loading = c_m * c_G * K_ads * RH / ((1 - K_ads * RH)*(1+(c_G-1)*K_ads*RH))
    adsorption_heat_2 = -49  # kJ/mol
    return equilibrium_loading, adsorption_heat_2

def calculate_gas_heat_capacity():
    # Example function to calculate gas heat capacity
    # This is a placeholder and should be replaced with actual calculations or data
    Cp_g = 29.19  # J/(mol*K) for air at room temperature
    return Cp_g

def heat_transfer_coefficient():
    heat_transfer_coefficient = 3
    wall_heat_transfer_coefficient = 26  # W/(m^2*K) Example value for wall heat transfer coefficient
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

def mass_balance_error(F_result, P_result, T_result, y1_result, n1_result, time, bed_properties, column_grid):
    bed_voidage = bed_properties["bed_voidage"]
    column_area = bed_properties["column_area"]
    R = bed_properties["R"]
    z = column_grid["xCentres"]
    mole_in = scipy.integrate.trapezoid(F_result[0], time)
    mole_out = scipy.integrate.trapezoid(F_result[-1], time)
    mole_acc == (scipy.integrate.trapezoid(
        (bed_voidage* column_area * P_result[-1,:] * y1_result[-1,:] / (R * T_result[-1,:])
        + (1-bed_voidage)*column_area * n1_result[-1,:]), z)
        -scipy.integrate.trapezoid(
        (bed_voidage* column_area * P_result[0,:] * y1_result[0,:] / (R * T_result[0,:])
        + (1-bed_voidage)*column_area * n1_result[0,:]), z)
        )
        # check indexing / orientation of array
    mass_balance_error = np.abs(mole_in - mole_out - mol_acc)/mole_acc*100
    return mass_balance_error
