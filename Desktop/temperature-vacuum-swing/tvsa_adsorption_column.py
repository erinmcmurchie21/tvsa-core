#Import modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from additional_functions import create_non_uniform_grid, adsorption_isotherm_1, adsorption_isotherm_2, quadratic_extrapolation, quadratic_extrapolation_derivative, calculate_gas_heat_capacity


mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

### Split results vector from previous time step

def data_prep(results_vector, num_cells):
    """
    Split the results vector into individual variables.
    Four component system where y1 + y2 + y3 + y4 = 1; two adsorbing components with (CO2 and H2O)
    Vector contains P, T, Tw, y1, y2, y3, n1, n2, F
    """

    P = results_vector[:num_cells]  # Pressure
    T = results_vector[num_cells:2*num_cells]  # Temperature
    Tw = results_vector[2*num_cells:3*num_cells]  # Wall temperature
    y1 = results_vector[3*num_cells:4*num_cells]  # Mole fraction of component 1 (e.g., CO2)
    y2 = results_vector[4*num_cells:5*num_cells]  # Mole fraction of component 2 (e.g., H2O)
    y3 = results_vector[5*num_cells:6*num_cells]  # Mole fraction of component 3 (e.g., N2)
    n1 = results_vector[6*num_cells:7*num_cells]  # Concentration of component 1 (e.g., CO2)
    n2 = results_vector[7*num_cells:8*num_cells]  # Concentration of component 2 (e.g., H2O)
    F = results_vector[8*num_cells:8*num_cells+4]  # Additional variables (e.g., flow rates, mass balances)

    return P, T, Tw, y1, y2, y3, n1, n2, F

def inlet_boundary_conditions(P, T, Tw, y1, y2, y3, column_grid, bed_properties, inlet_values):
    """
    Apply boundary conditions to the column model.
    This function modifies the state variables based on the inlet and outlet conditions.
    """
    # Inlet conditions
    if inlet_values["inlet_type"] == "mass_flow":
        y1_inlet = float(inlet_values["y1_feed_value"])  # Example value for mole fraction of component 1
        y2_inlet = float(inlet_values["y2_feed_value"])
        y3_inlet = float(inlet_values["y3_feed_value"])
        rho_gas = inlet_values["rho_gas"]  # Example value for feed density in kg/m^3
        volumetric_flow_rate_inlet = inlet_values["feed_mass_flow"] / rho_gas
        v_inlet = volumetric_flow_rate_inlet / bed_properties["column_area"]  # Superficial velocity
        dPdz_inlet = (1.75 * (1 - bed_properties["bed_voidage"]) * inlet_values["rho_gas"] * v_inlet**2 / 
                     (bed_properties["bed_voidage"]**3 * bed_properties["particle_diameter"]) + 
                     150 * inlet_values["mu"] * (1 - bed_properties["bed_voidage"])**2 * v_inlet / 
                     (bed_properties["bed_voidage"]**3 * bed_properties["particle_diameter"]**2))
        #use dPdZ to find P_inlet_
        P_inlet = P[0] - dPdz_inlet * (column_grid["xWalls"][int(column_grid["nGhost"])] - column_grid["xCentres"][int(column_grid["nGhost"])])
        T_inlet = inlet_values["feed_temperature"]  # Example value for feed temperature in Kelvin
        Tw_inlet = quadratic_extrapolation_derivative(column_grid["xCentres"][column_grid["nGhost"]], Tw[0], column_grid["xCentres"][column_grid["nGhost"]+1], Tw[1],
                                               column_grid["xCentres"][column_grid["nGhost"]+2], Tw[2], column_grid["xWalls"][column_grid["nGhost"]])
        dTwdz_inlet=0

    elif inlet_values["inlet_type"] == "closed":
        dPdz_inlet = 0
        P_inlet = P[0]
        y1_inlet = quadratic_extrapolation_derivative(column_grid["xCentres"][column_grid["nGhost"]], y1[0], column_grid["xCentres"][column_grid["nGhost"]+1], y1[1],
                                               column_grid["xCentres"][column_grid["nGhost"]+2], y1[2], column_grid["xWalls"][column_grid["nGhost"]])
        y2_inlet = quadratic_extrapolation_derivative(column_grid["xCentres"][column_grid["nGhost"]], y2[0], column_grid["xCentres"][column_grid["nGhost"]+1], y2[1],
                                               column_grid["xCentres"][column_grid["nGhost"]+2], y2[2], column_grid["xWalls"][column_grid["nGhost"]])
        y3_inlet = quadratic_extrapolation_derivative(column_grid["xCentres"][column_grid["nGhost"]], y3[0], column_grid["xCentres"][column_grid["nGhost"]+1], y3[1],
                                               column_grid["xCentres"][column_grid["nGhost"]+2], y3[2], column_grid["xWalls"][column_grid["nGhost"]])
        v_inlet = 0
        T_inlet = quadratic_extrapolation_derivative(column_grid["xCentres"][column_grid["nGhost"]], T[0], column_grid["xCentres"][column_grid["nGhost"]+1], T[1],
                                               column_grid["xCentres"][column_grid["nGhost"]+2], T[2], column_grid["xWalls"][column_grid["nGhost"]])
        Tw_inlet = quadratic_extrapolation_derivative(column_grid["xCentres"][column_grid["nGhost"]], Tw[0], column_grid["xCentres"][column_grid["nGhost"]+1], Tw[1],
                                               column_grid["xCentres"][column_grid["nGhost"]+2], Tw[2], column_grid["xWalls"][column_grid["nGhost"]])
        dTwdz_inlet=0
    
        return P_inlet, T_inlet, Tw_inlet, y1_inlet, y2_inlet, y3_inlet, v_inlet, dPdz_inlet, dTwdz_inlet

def outlet_boundary_conditions(P, T, Tw, y1, y2, y3, column_grid, bed_properties, outlet_values):

    if outlet_values["outlet_type"] == "pressure":
        P_outlet = outlet_values["outlet_pressure"]
        y1_outlet = quadratic_extrapolation(column_grid["xCentres"][-(column_grid["nGhost"]+1)], y1[-1], column_grid["xCentres"][-(column_grid["nGhost"]+2)], y1[-2],
                                               column_grid["xCentres"][-(column_grid["nGhost"]+3)], y1[-3], column_grid["xWalls"][-(column_grid["nGhost"]+1)])
        y2_outlet = quadratic_extrapolation(column_grid["xCentres"][-(column_grid["nGhost"]+1)], y2[-1], column_grid["xCentres"][-(column_grid["nGhost"]+2)], y2[-2],
                                               column_grid["xCentres"][-(column_grid["nGhost"]+3)], y2[-3], column_grid["xWalls"][-(column_grid["nGhost"]+1)])
        y3_outlet = quadratic_extrapolation(column_grid["xCentres"][-(column_grid["nGhost"]+1)], y3[-1], column_grid["xCentres"][-(column_grid["nGhost"]+2)], y3[-2],
                                               column_grid["xCentres"][-(column_grid["nGhost"]+3)], y3[-3], column_grid["xWalls"][-(column_grid["nGhost"]+1)])
        T_outlet = quadratic_extrapolation_derivative(column_grid["xCentres"][-(column_grid["nGhost"]+1)], T[-1], column_grid["xCentres"][-(column_grid["nGhost"]+2)], T[-2],
                                               column_grid["xCentres"][-(column_grid["nGhost"]+3)], T[-3], column_grid["xWalls"][-(column_grid["nGhost"]+1)])
        Tw_outlet = quadratic_extrapolation_derivative(column_grid["xCentres"][-(column_grid["nGhost"]+1)], Tw[-1], column_grid["xCentres"][-(column_grid["nGhost"]+2)], Tw[-2],
                                               column_grid["xCentres"][-(column_grid["nGhost"]+3)], Tw[-3], column_grid["xWalls"][-(column_grid["nGhost"]+1)])
        rho_gas_outlet = P_outlet / bed_properties["R"] / T_outlet  # Assuming ideal gas law for density calculation
        avg_density_outlet_ = rho_gas_outlet * (bed_properties["MW_1"] * y1_outlet + bed_properties["MW_2"]*(y2_outlet) + 
                                              bed_properties["MW_3"]*(y3_outlet)+ bed_properties["MW_4"]*(1-y1_outlet-y2_outlet-y3_outlet))
        dPdz_outlet = (P_outlet - P[-1]) / (column_grid["xWalls"][-1] - column_grid["xCentres"][-1])
            #### how do I calculate the desnity in the ergun equation at the outlet??
        a = 1.75 * (1- bed_properties["bed_voidage"]) * avg_density_outlet_ / (bed_properties["bed_voidage"]**3 * bed_properties["particle_diameter"])
        b = 12 * outlet_values["mu"] * (1-bed_properties["bed_voidage"])**2 / (bed_properties["bed_voidage"]**3 * bed_properties["particle_diameter"]**2)
        c = np.abs(dPdz_outlet)
        v_outlet = - np.sign(dPdz_outlet) * ((-b + np.sqrt(b**2 + 4*a*c)) / (2*a))
        dTwdz_outlet=0

    elif outlet_values["outlet_type"] == "closed":
        P_outlet = P[-(column_grid["nGhost"]+1)]
        y1_outlet = quadratic_extrapolation_derivative(column_grid["xCentres"][-(column_grid["nGhost"]+1)], y1[-1], column_grid["xCentres"][-(column_grid["nGhost"]+2)], y1[-2],
                                               column_grid["xCentres"][-(column_grid["nGhost"]+3)], y1[-3], column_grid["xWalls"][-(column_grid["nGhost"]+1)])
        y2_outlet = quadratic_extrapolation_derivative(column_grid["xCentres"][-(column_grid["nGhost"]+1)], y2[-1], column_grid["xCentres"][-(column_grid["nGhost"]+2)], y2[-2],
                                               column_grid["xCentres"][-(column_grid["nGhost"]+3)], y2[-3], column_grid["xWalls"][-(column_grid["nGhost"]+1)])
        y3_outlet = quadratic_extrapolation_derivative(column_grid["xCentres"][-(column_grid["nGhost"]+1)], y3[-1], column_grid["xCentres"][-(column_grid["nGhost"]+2)], y3[-2],
                                               column_grid["xCentres"][-(column_grid["nGhost"]+3)], y3[-3], column_grid["xWalls"][-(column_grid["nGhost"]+1)])
        T_outlet = quadratic_extrapolation_derivative(column_grid["xCentres"][-(column_grid["nGhost"]+1)], T[-1], column_grid["xCentres"][-(column_grid["nGhost"]+2)], T[-2],
                                               column_grid["xCentres"][-(column_grid["nGhost"]+3)], T[-3], column_grid["xWalls"][-(column_grid["nGhost"]+1)])
        Tw_outlet = quadratic_extrapolation_derivative(column_grid["xCentres"][-(column_grid["nGhost"]+1)], Tw[-1], column_grid["xCentres"][-(column_grid["nGhost"]+2)], Tw[-2],
                                               column_grid["xCentres"][-(column_grid["nGhost"]+3)], Tw[-3], column_grid["xWalls"][-(column_grid["nGhost"]+1)])
        v_outlet=0
        dPdz_outlet=0
        dTwdz_outlet=0

    elif outlet_values["outlet_type"] == "mass_flow":
        P_outlet = P[-1]

    return P_outlet, T_outlet, Tw_outlet, y1_outlet, y2_outlet, y3_outlet, v_outlet, dPdz_outlet, dTwdz_outlet

def ghost_cell_calculations(P, T, Tw, y1, y2, y3, v, 
                            P_inlet, P_outlet, T_inlet, T_outlet, Tw_inlet, Tw_outlet, y1_inlet, y1_outlet, y2_inlet, y2_outlet, 
                            y3_inlet, y3_outlet, column_grid):
    """
    Calculate ghost cell values for the column model.
    This function computes the ghost cell values based on the boundary conditions.
    """
    num_cells = column_grid["num_cells"]
    
    # Example ghost cell calculations (to be implemented)
    P_ghost_start = P[0] + (column_grid["xCentres"][0] - column_grid["xCentres"][1]) * (P[0] - P_inlet) / (column_grid["xCentres"][column_grid["nGhost"]] - column_grid["xWalls"][column_grid["nGhost"]])
    P_ghost_end = P[-1] - (column_grid["xCentres"][-1] - column_grid["xCentres"][-column_grid["nGhost"]-1]) * (P[-1] - P_outlet) / (column_grid["xCentres"][-(column_grid["nGhost"]+1)] - column_grid["xWalls"][-(column_grid["nGhost"]+1)]) 
    P_all = np.concatenate(np.array([P_ghost_start]), P, np.array([P_ghost_end]))

    y1_ghost_start = quadratic_extrapolation(column_grid["xWalls"][column_grid["nGhost"]], y1_inlet, column_grid["xCentres"][column_grid["nGhost"]], y1[0], column_grid["xCentres"][column_grid["nGhost"]+1], y1[1], column_grid["xCentres"][0])
    y1_ghost_end = quadratic_extrapolation(column_grid["xWalls"][-(column_grid["nGhost"]+1)], y1_outlet, column_grid["xCentres"][-(column_grid["nGhost"]+1)], y1[-1], column_grid["xCentres"][-(column_grid["nGhost"]+2)], y1[-2], column_grid["xCentres"][-1])
    y1_all = np.concatenate((np.array([y1_ghost_start]), y1,np.array([y1_ghost_end]))) 

    y2_ghost_start = quadratic_extrapolation(column_grid["xWalls"][column_grid["nGhost"]], y2_inlet, column_grid["xCentres"][column_grid["nGhost"]], y2[0], column_grid["xCentres"][column_grid["nGhost"]+1], y2[1], column_grid["xCentres"][0])
    y2_ghost_end = quadratic_extrapolation(column_grid["xWalls"][-(column_grid["nGhost"]+1)], y2_outlet, column_grid["xCentres"][-(column_grid["nGhost"]+1)], y2[-1], column_grid["xCentres"][-(column_grid["nGhost"]+2)], y2[-2], column_grid["xCentres"][-1])
    y2_all = np.concatenate((np.array([y2_ghost_start]), y2,np.array([y2_ghost_end]))) 

    y3_ghost_start = quadratic_extrapolation(column_grid["xWalls"][column_grid["nGhost"]], y3_inlet, column_grid["xCentres"][column_grid["nGhost"]], y3[0], column_grid["xCentres"][column_grid["nGhost"]+1], y3[1], column_grid["xCentres"][0])
    y3_ghost_end = quadratic_extrapolation(column_grid["xWalls"][-(column_grid["nGhost"]+1)], y3_outlet, column_grid["xCentres"][-(column_grid["nGhost"]+1)], y3[-1], column_grid["xCentres"][-(column_grid["nGhost"]+2)], y3[-2], column_grid["xCentres"][-1])
    y3_all = np.concatenate((np.array([y3_ghost_start]), y3,np.array([y3_ghost_end]))) 
    
    T_ghost_start = quadratic_extrapolation(column_grid["xWalls"][column_grid["nGhost"]], T_inlet, column_grid["xCentres"][column_grid["nGhost"]], T[0], column_grid["xCentres"][column_grid["nGhost"]+1], T[1], column_grid["xCentres"][0])
    T_ghost_end = quadratic_extrapolation(column_grid["xWalls"][-(column_grid["nGhost"]+1)], T_outlet, column_grid["xCentres"][-(column_grid["nGhost"]+1)], T[-1], column_grid["xCentres"][-(column_grid["nGhost"]+2)], T[-2], column_grid["xCentres"][-1])
    T_all = np.concatenate((np.array([T_ghost_start]), T,np.array([T_ghost_end])))

    Tw_ghost_start = Tw[0] + (column_grid["xCentres"][0] - column_grid["xCentres"][1]) * (Tw[0] - Tw_inlet) / (column_grid["xCentres"][column_grid["nGhost"]] - column_grid["xWalls"][column_grid["nGhost"]])
    Tw_ghost_end = Tw[-1] - (column_grid["xCentres"][-1] - column_grid["xCentres"][-column_grid["nGhost"]-1]) * (Tw[-1] - Tw_outlet) / (column_grid["xCentres"][-(column_grid["nGhost"]+1)] - column_grid["xWalls"][-(column_grid["nGhost"]+1)]) 
    Tw_all = np.concatenate(np.array([Tw_ghost_start]), Tw, np.array([Tw_ghost_end]))

    return (P_all, T_all, Tw_all, y1_all, y2_all, y3_all)

def calculate_internal_wall_values(P_all, T_all, Tw_all, y1_all, y2_all, y3_all, 
                                   P_inlet, P_outlet, T_inlet, T_outlet, Tw_inlet, Tw_outlet, y1_inlet, y1_outlet, y2_inlet, y2_outlet, 
                                    y3_inlet, y3_outlet, v_inlet, v_outlet, dPdz_inlet, dPdz_outlet, dTwdz_inlet, dTwdz_outlet, bed_properties, column_grid):

    epsilon = 1.0e-10
    Nx = int(column_grid["num_cells"])
    #y1 vector at cell walls, from van leer flux limiter function
    R_r = (column_grid["deltaZ"][2:Nx+2] + column_grid["deltaZ"][1:Nx+1]) / column_grid["deltaZ"][1:Nx+1]
    r_r1 = ((y1_all[1:Nx+1] - y1_all[:Nx]) + epsilon)/((y1_all[2:Nx+2]-y1_all[1:Nx+1])+ epsilon)*(column_grid["deltaZ"][2:Nx+2]+column_grid["deltaZ"][1:Nx+1])/(column_grid["deltaZ"][1:Nx+1] + column_grid["deltaZ"][0:Nx])
    modified_van_leer1 = (0.5 * R_r * r_r1 + 0.5 * R_r * abs(r_r1))/(R_r + r_r1 - 1 )
    flux_limiter1 = modified_van_leer1 / R_r
    y1_walls = y1_all[1:Nx+1] + flux_limiter1 * (y1_all[2:Nx+2]-y1_all[1:Nx+1])
    y1_walls[-1] = y1_outlet
    y1_walls = np.concatenate((np.array([y1_inlet]), y1_walls))

    #y2 vector at cell walls, from van leer flux limiter function
    r_r2 = ((y2_all[1:Nx+1] - y2_all[:Nx]) + epsilon)/((y2_all[2:Nx+2]-y2_all[1:Nx+1])+ epsilon)*(column_grid["deltaZ"][2:Nx+2]+column_grid["deltaZ"][1:Nx+1])/(column_grid["deltaZ"][1:Nx+1] + column_grid["deltaZ"][0:Nx])
    modified_van_leer2 = (0.5 * R_r * r_r2 + 0.5 * R_r * abs(r_r2))/(R_r + r_r2 - 1 )
    flux_limiter2 = modified_van_leer2 / R_r
    y2_walls = y2_all[1:Nx+1] + flux_limiter2 * (y2_all[2:Nx+2]-y2_all[1:Nx+1])
    y2_walls[-1] = y2_outlet
    y2_walls = np.concatenate((np.array([y2_inlet]), y2_walls))

    #y3 vector at cell walls, from van leer flux limiter function
    r_r3 = ((y3_all[1:Nx+1] - y3_all[:Nx]) + epsilon)/((y3_all[2:Nx+2]-y3_all[1:Nx+1])+ epsilon)*(column_grid["deltaZ"][2:Nx+2]+column_grid["deltaZ"][1:Nx+1])/(column_grid["deltaZ"][1:Nx+1] + column_grid["deltaZ"][0:Nx])
    modified_van_leer3 = (0.5 * R_r * r_r3 + 0.5 * R_r * abs(r_r3))/(R_r + r_r3 - 1 )
    flux_limiter3 = modified_van_leer3 / R_r
    y3_walls = y3_all[1:Nx+1] + flux_limiter3 * (y3_all[2:Nx+2]-y3_all[1:Nx+1])
    y3_walls[-1] = y3_outlet
    y3_walls = np.concatenate((np.array([y3_inlet]), y3_walls))

    r_r_T = ((T_all[1:Nx+1] - T_all[:Nx]) + epsilon)/((T_all[2:Nx+2]-T_all[1:Nx+1])+ epsilon)*(column_grid["deltaZ"][2:Nx+2]+column_grid["deltaZ"][1:Nx+1])/(column_grid["deltaZ"][1:Nx+1] + column_grid["deltaZ"][0:Nx])
    modified_van_leer_T = (0.5 * R_r * r_r_T + 0.5 * R_r * abs(r_r_T))/(R_r + r_r_T - 1 )
    flux_limiter_T = modified_van_leer_T / R_r
    T_walls = T_all[1:Nx+1] + flux_limiter_T * (T_all[2:Nx+2]-T_all[1:Nx+1])
    T_walls[-1] = T_outlet
    T_walls = np.concatenate((np.array([T_inlet]), T_walls))

    dTdz_walls = np.array((T_all[1:Nx+2]- T_all[0:Nx+1]) / (column_grid["xCentres"][1:Nx+2] - column_grid["xCentres"][0:Nx+1]))

    #calculate dP/dz at internal cell walls by linear interpolation
    dPdz_walls = np.array((P_all[1:Nx+2]- P_all[0:Nx+1]) / (column_grid["xCentres"][1:Nx+2] - column_grid["xCentres"][0:Nx+1]))
    dPdz_walls[0] = dPdz_inlet  # set inlet pressure gradient
    dPdz_walls[-1] = dPdz_outlet

    #calculate P at cell walls by interpolation
    P_walls = np.array(P_all[0:Nx+1] + dPdz_walls * (column_grid["deltaZ"][0:Nx+1]/2))
    P_walls[0] = P_inlet  # set inlet pressure
    P_walls[-1] = P_outlet  # set outlet pressure

    #calculate dTw/dz at internal cell walls by linear interpolation
    dTwdz_walls = np.array((Tw_all[1:Nx+2]- Tw_all[0:Nx+1]) / (column_grid["xCentres"][1:Nx+2] - column_grid["xCentres"][0:Nx+1]))
    dTwdz_walls[0] = dTwdz_inlet  # set inlet pressure gradient
    dTwdz_walls[-1] = dTwdz_outlet

    #calculate Tw at cell walls by interpolation
    Tw_walls = np.array(Tw_all[0:Nx+1] + dTwdz_walls * (column_grid["deltaZ"][0:Nx+1]/2))
    Tw_walls[0] = Tw_inlet  # set inlet pressure
    Tw_walls[-1] = Tw_outlet  # set outlet pressure


    rho_gas_walls = P_walls / bed_properties["R"] / T_walls  # Assuming ideal gas law for density calculation
    avg_density_walls = rho_gas_walls * (bed_properties["MW_1"] * y1_walls + bed_properties["MW_2"]*(y2_walls) + 
                                              bed_properties["MW_3"]*(y3_walls)+ bed_properties["MW_4"]*(1-y1_walls-y2_walls-y3_walls))

    a = 1.75 * (1- bed_properties["bed_voidage"]) * avg_density_walls / (bed_properties["bed_voidage"]**3 * bed_properties["particle_diameter"])
    b = 12 * bed_properties["mu"] * (1-bed_properties["bed_voidage"])**2 / (bed_properties["bed_voidage"]**3 * bed_properties["particle_diameter"]**2)
    c = np.abs(dPdz_walls[:])
    dominant = b**2+4*a*c
        
    if np.any(dominant < 0):
        raise ValueError("Negative value under square root in velocity calculation. Check your inputs and boundary conditions.")
    v_walls = np.array((-b + np.sqrt(dominant)) / (2*a)) # should have N + 1 values 
    v_walls = np.multiply(-np.sign(dPdz_walls),v_walls)  # make sure velocity is in the correct direction
    v_walls[0] = v_inlet  # set inlet velocity
    v_walls[-1] = v_outlet # set outlet velocity

    
    return P_walls, T_walls, Tw_walls, y1_walls, y2_walls, y3_walls, v_walls, dTdz_walls

def ODE_calculations(t, results_vector, column_grid, bed_properties, inlet_values, outlet_values):
    """
    Calculate the ODEs for the adsorption column model.
    This function computes the derivatives of the state variables.
    """
    num_cells = column_grid["num_cells"]
    # Split results vector into individual variables
    P, T, Tw, y1, y2, y3, n1, n2, F = data_prep(results_vector, column_grid["num_cells"])
    
    # Apply inlet boundary conditions
    P_inlet, T_inlet, Tw_inlet, y1_inlet, y2_inlet, y3_inlet, v_inlet, dPdz_inlet, dTwdz_inlet = inlet_boundary_conditions(P, T, Tw, y1, y2, y3, column_grid, bed_properties, inlet_values)

    # Apply outlet boundary conditions
    P_outlet, T_outlet, Tw_outlet, y1_outlet, y2_outlet, y3_outlet, v_outlet, dPdz_outlet, dTwdz_outlet = outlet_boundary_conditions(P, T, Tw, y1, y2, y3, column_grid, bed_properties, outlet_values)
    
    # Calculate ghost cell values
    P_all, T_all, Tw_all, y1_all, y2_all, y3_all = ghost_cell_calculations(P, T, Tw, y1, y2, y3, P_inlet, P_outlet, T_inlet, T_outlet, Tw_inlet, 
                                                                           Tw_outlet, y1_inlet, y1_outlet, y2_inlet, y2_outlet, y3_inlet, y3_outlet, column_grid)

    # Calculate internal wall values
    P_walls, T_walls, Tw_walls, y1_walls, y2_walls, y3_walls, v_walls, dTdz_walls = calculate_internal_wall_values(
        P_all, T_all, Tw_all, y1_all, y2_all, y3_all,
        P_inlet, P_outlet, T_inlet, T_outlet, Tw_inlet, Tw_outlet,
        y1_inlet, y1_outlet, y2_inlet, y2_outlet,
        y3_inlet, y3_outlet, v_inlet, v_outlet,
        dPdz_inlet, dPdz_outlet, dTwdz_inlet, dTwdz_outlet,
        bed_properties, column_grid
    )
    #ODE calculations (to be implemented)
    k1 = 30 # s-1 linear driving force mass transfer constant
    dn1dt = k1*(adsorption_isotherm_1(P, T, y1)-n1)
            
        #updating central values
    k2 = 30 # s-1 linear driving force mass transfer constant
    dn2dt = k2*(adsorption_isotherm_2(P, T, y2)-n2)

    # pressure derivative
    dPdt = np.zeros(num_cells)

    # temperature derivative
    dTdt = np.zeros(num_cells)
    coeff_1 = ((1- bed_properties["bed_voidage"])/bed_properties["bed_voidage"] * (bed_properties["solid_heat_capacity"] * 
                            bed_properties["sorbent_mass"] + (bed_properties["heat_capacity_1"] * n1 + bed_properties["heat_capacity_2"] * n2)))
    d2Tdz2 = 1/(column_grid["deltaZ"][1:num_cells+1]) *(dTdz_walls[1:num_cells+1] - dTdz_walls[:num_cells])
    K_z = 0
    Cp_g = calculate_gas_heat_capacity()
    dTdt = 1 / coeff_1 * (K_z/bed_properties["bed_voidage"]*d2Tdz2 - Cp_g/(bed_properties["R"]*column_grid["deltaZ"])*
                          (v_walls[1:num_cells+1]*P_walls[1:num_cells+1] - v_walls[:num_cells]*P_walls[:num_cells]) - 
                          Cp_g/bed_properties["R"] *dPdt - (1-bed_properties["bed_voidage"])/bed_properties["bed_voidage"] * T
                            (bed_properties["heat_capacity_1"] * n1 + bed_properties["heat_capacity_2"] * n2) + 
                            (1-bed_properties["bed_voidage"])/bed_properties["bed_voidage"]*(bed_properties["adsorption_heat_1"] * dn1dt +
                            bed_properties["adsorption_heat_2"] * dn2dt) -
                              2* bed_properties["heat_transfer_coefficient"] *(T-Tw)/(bed_properties["bed_voidage"]*bed_properties["inner_bed_diameter"]) 
                          )
    
    
    # wall temperature derivative
    dTwdt = np.zeros(num_cells)


    dy1dt = np.zeros(num_cells)
    dy2dt = np.zeros(num_cells)
    dy3dt = np.zeros(num_cells)
    
    dF1dt = bed_properties["bed_voidage"] * bed_properties["column_area"] / (bed_properties["R"] * T_walls[0]) * v_walls[0]*P_walls[0] * y1_walls[0]
    dF2dt = bed_properties["bed_voidage"] * bed_properties["column_area"] / (bed_properties["R"] * T_walls[0]) * v_walls[0]*P_walls[0] * y2_walls[0]
    dF3dt = bed_properties["bed_voidage"] * bed_properties["column_area"] / (bed_properties["R"] * T_walls[0]) * v_walls[0]*P_walls[0] * y3_walls[0]
    dF4dt = bed_properties["bed_voidage"] * bed_properties["column_area"] / (bed_properties["R"] * T_walls[0]) * v_walls[0]*P_walls[0] * (1 - y1_walls[0]-y2_walls[0]-y3_walls[0])
    dF5dt = bed_properties["bed_voidage"] * bed_properties["column_area"] / (bed_properties["R"] * T_walls[-1]) * v_walls[-1]*P_walls[-1] * y1_walls[-1]
    dF6dt = bed_properties["bed_voidage"] * bed_properties["column_area"] / (bed_properties["R"] * T_walls[-1]) * v_walls[-1]*P_walls[-1] * y2_walls[-1]
    dF7dt = bed_properties["bed_voidage"] * bed_properties["column_area"] / (bed_properties["R"] * T_walls[-1]) * v_walls[-1]*P_walls[-1] * y3_walls[-1]
    dF8dt = bed_properties["bed_voidage"] * bed_properties["column_area"] / (bed_properties["R"] * T_walls[-1]) * v_walls[-1]*P_walls[-1] * (1 - y1_walls[-1]-y2_walls[-1]-y3_walls[-1])
    dFdt = np.array([dF1dt, dF2dt, dF3dt, dF4dt, dF5dt, dF6dt, dF7dt, dF8dt])
    # Combine derivatives into a single vector
    derivatives = np.concatenate([dPdt, dTdt, dTwdt, dy1dt, dy2dt, dy3dt, dn1dt, dn2dt, dFdt])
    
    return derivatives