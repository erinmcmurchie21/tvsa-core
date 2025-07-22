"""
This is the code which describes the PDEs for the adsorption column.
The input vector is split into the vectors for each state variable, and then the inlet and outlet boundary conditions are applied.
THe bounary conditions are used to calculate the values in the ghost cells, and then the value for the state variables are 
calculated at each wall position. The wall values are then used to calculate the differential term which is explorted to the
timestepper to be solved.
"""

#Import modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import additional_functions as func

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

### Split results vector from previous time step #######

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
    F = results_vector[8*num_cells:8*num_cells+8]  # Additional variables (e.g., flow rates, mass balances)
    E = results_vector[8*num_cells+8:]  # Additional energy variables

    return P, T, Tw, y1, y2, y3, n1, n2, F

def inlet_boundary_conditions(P, T, Tw, y1, y2, y3, column_grid, bed_properties, inlet_values):
    """
    Apply boundary conditions to the column model.
    This function finds the values of the state variables at the inlet wall where z=0.
    """
    # Inlet conditions
    if inlet_values["inlet_type"] == "mass_flow":
        # Calculate gas properties at the inlet
        rho_gas = func.calculate_gas_density()  # Example value for feed density in kg/m^3
        mu = func.calculate_gas_viscosity()
        D_l = func.calculate_axial_dispersion_coefficient(bed_properties, inlet_values)
        volumetric_flow_rate_inlet = inlet_values["feed_mass_flow"] / rho_gas
        v_inlet = volumetric_flow_rate_inlet / bed_properties["column_area"]  # Superficial velocity
        Cp_g = func.calculate_gas_heat_capacity()  # Example function to calculate gas specific heat capacity
        thermal_diffusivity = func.calculate_gas_thermal_conductivity() / (Cp_g * rho_gas)

        # Calculate pressure drop at the inlet using Ergun equation
        dPdz_inlet = (1.75 * (1 - bed_properties["bed_voidage"]) * rho_gas * v_inlet**2 / 
                     (bed_properties["bed_voidage"]**3 * bed_properties["particle_diameter"]) + 
                     150 * mu * (1 - bed_properties["bed_voidage"])**2 * v_inlet / 
                     (bed_properties["bed_voidage"]**3 * bed_properties["particle_diameter"]**2))
        
        # Use dPdZ to find P_inlet_
        P_inlet = P[0] - dPdz_inlet * (column_grid["xWalls"][int(column_grid["nGhost"])] - column_grid["xCentres"][int(column_grid["nGhost"])])
    
        # Calculating the inlet concentration values
        # dyidz = a * (b-yi(x=0))
        y1_inlet = func.quadratic_extrapolation_derivative_nonzero(column_grid["xCentres"][column_grid["nGhost"]], y1[0], column_grid["xCentres"][column_grid["nGhost"]+1], y1[1],
                                                column_grid["xWalls"][column_grid["nGhost"]], -np.abs(v_inlet / D_l), inlet_values["y1_feed_value"]) 
        y2_inlet = func.quadratic_extrapolation_derivative_nonzero(column_grid["xCentres"][column_grid["nGhost"]], y2[0], column_grid["xCentres"][column_grid["nGhost"]+1], y2[1],
                                                column_grid["xWalls"][column_grid["nGhost"]], -np.abs(v_inlet / D_l), inlet_values["y2_feed_value"])
        y3_inlet = func.quadratic_extrapolation_derivative_nonzero(column_grid["xCentres"][column_grid["nGhost"]], y3[0], column_grid["xCentres"][column_grid["nGhost"]+1], y3[1],
                                                column_grid["xWalls"][column_grid["nGhost"]], -np.abs(v_inlet / D_l), inlet_values["y3_feed_value"])
        
        # Calculating the inlet temperature values
        # dTdz = a * (b-T[inlet]) = -v_inlet * Pe_h * (T_feed - T_inlet)
        Pe_h = v_inlet * bed_properties["inner_bed_diameter"]/thermal_diffusivity  # Peclet number for heat transfer
        a = -v_inlet * Pe_h
        b = inlet_values["feed_temperature"]
        T_inlet = func.quadratic_extrapolation_derivative_nonzero(column_grid["xCentres"][column_grid["nGhost"]], T[0], 
            column_grid["xCentres"][column_grid["nGhost"]+1], T[1], 
            column_grid["xWalls"][column_grid["nGhost"]], a, b)
        Tw_inlet = Tw[0]
        dTwdz_inlet=0

    elif inlet_values["inlet_type"] == "closed":
        dPdz_inlet = 0
        P_inlet = P[0]
        y1_inlet = func.quadratic_extrapolation_derivative(column_grid["xCentres"][column_grid["nGhost"]], y1[0], column_grid["xCentres"][column_grid["nGhost"]+1], y1[1],
                                                column_grid["xWalls"][column_grid["nGhost"]])
        y2_inlet = func.quadratic_extrapolation_derivative(column_grid["xCentres"][column_grid["nGhost"]], y2[0], column_grid["xCentres"][column_grid["nGhost"]+1], y2[1],
                                                column_grid["xWalls"][column_grid["nGhost"]])
        y3_inlet = func.quadratic_extrapolation_derivative(column_grid["xCentres"][column_grid["nGhost"]], y3[0], column_grid["xCentres"][column_grid["nGhost"]+1], y3[1],
                                                column_grid["xWalls"][column_grid["nGhost"]])
        v_inlet = 0
        T_inlet = func.quadratic_extrapolation_derivative(column_grid["xCentres"][column_grid["nGhost"]], T[0], column_grid["xCentres"][column_grid["nGhost"]+1], T[1],
                                               column_grid["xCentres"][column_grid["nGhost"]+2], T[2], column_grid["xWalls"][column_grid["nGhost"]])
        Tw_inlet = Tw[0]                            
        dTwdz_inlet=0
    
    return P_inlet, T_inlet, Tw_inlet, y1_inlet, y2_inlet, y3_inlet, v_inlet, dPdz_inlet, dTwdz_inlet

def outlet_boundary_conditions(P, T, Tw, y1, y2, y3, column_grid, bed_properties, outlet_values):

    if outlet_values["outlet_type"] == "pressure":
        P_outlet = outlet_values["outlet_pressure"]
        y1_outlet = func.quadratic_extrapolation(column_grid["xCentres"][-(column_grid["nGhost"]+1)], y1[-1], column_grid["xCentres"][-(column_grid["nGhost"]+2)], y1[-2],
                                               column_grid["xCentres"][-(column_grid["nGhost"]+3)], y1[-3], column_grid["xWalls"][-(column_grid["nGhost"]+1)])
        y2_outlet = func.quadratic_extrapolation(column_grid["xCentres"][-(column_grid["nGhost"]+1)], y2[-1], column_grid["xCentres"][-(column_grid["nGhost"]+2)], y2[-2],
                                               column_grid["xCentres"][-(column_grid["nGhost"]+3)], y2[-3], column_grid["xWalls"][-(column_grid["nGhost"]+1)])
        y3_outlet = func.quadratic_extrapolation(column_grid["xCentres"][-(column_grid["nGhost"]+1)], y3[-1], column_grid["xCentres"][-(column_grid["nGhost"]+2)], y3[-2],
                                               column_grid["xCentres"][-(column_grid["nGhost"]+3)], y3[-3], column_grid["xWalls"][-(column_grid["nGhost"]+1)])
        T_outlet = func.quadratic_extrapolation_derivative(column_grid["xCentres"][-(column_grid["nGhost"]+1)], T[-1], column_grid["xCentres"][-(column_grid["nGhost"]+2)], T[-2],
                                               column_grid["xWalls"][-(column_grid["nGhost"]+1)])
        Tw_outlet = Tw[-1]
        mu = func.calculate_gas_viscosity()
        rho_gas_outlet = P_outlet / bed_properties["R"] / T_outlet  # Assuming ideal gas law for density calculation
        avg_density_outlet_ = rho_gas_outlet * (bed_properties["MW_1"] * y1_outlet + bed_properties["MW_2"]*(y2_outlet) + 
                                              bed_properties["MW_3"]*(y3_outlet)+ bed_properties["MW_4"]*(1-y1_outlet-y2_outlet-y3_outlet))
        dPdz_outlet = (P_outlet - P[-1]) / (column_grid["xWalls"][-1] - column_grid["xCentres"][-1])
            #### how do I calculate the desnity in the ergun equation at the outlet??
        a = 1.75 * (1- bed_properties["bed_voidage"]) * avg_density_outlet_ / (bed_properties["bed_voidage"]**3 * bed_properties["particle_diameter"])
        b = 12 * mu * (1-bed_properties["bed_voidage"])**2 / (bed_properties["bed_voidage"]**3 * bed_properties["particle_diameter"]**2)
        c = np.abs(dPdz_outlet)
        v_outlet = - np.sign(dPdz_outlet) * ((-b + np.sqrt(b**2 + 4*a*c)) / (2*a))
        dTwdz_outlet=0

    elif outlet_values["outlet_type"] == "closed":
        P_outlet = P[-(column_grid["nGhost"]+1)]
        y1_outlet = func.quadratic_extrapolation_derivative(column_grid["xCentres"][-(column_grid["nGhost"]+1)], y1[-1], column_grid["xCentres"][-(column_grid["nGhost"]+2)], y1[-2], column_grid["xWalls"][-(column_grid["nGhost"]+1)])
        y2_outlet = func.quadratic_extrapolation_derivative(column_grid["xCentres"][-(column_grid["nGhost"]+1)], y2[-1], column_grid["xCentres"][-(column_grid["nGhost"]+2)], y2[-2],
                                                column_grid["xWalls"][-(column_grid["nGhost"]+1)])
        y3_outlet = func.quadratic_extrapolation_derivative(column_grid["xCentres"][-(column_grid["nGhost"]+1)], y3[-1], column_grid["xCentres"][-(column_grid["nGhost"]+2)], y3[-2],
                                                column_grid["xWalls"][-(column_grid["nGhost"]+1)])
        T_outlet = func.quadratic_extrapolation_derivative(column_grid["xCentres"][-(column_grid["nGhost"]+1)], T[-1], column_grid["xCentres"][-(column_grid["nGhost"]+2)], T[-2],
                                                column_grid["xWalls"][-(column_grid["nGhost"]+1)])
        Tw_outlet = Tw[-1]
        v_outlet=0
        dPdz_outlet=0
        dTwdz_outlet=0

    elif outlet_values["outlet_type"] == "mass_flow":
        P_outlet = P[-1]

    return P_outlet, T_outlet, Tw_outlet, y1_outlet, y2_outlet, y3_outlet, v_outlet, dPdz_outlet, dTwdz_outlet

def ghost_cell_calculations(P, T, Tw, y1, y2, y3, 
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
    P_all = np.concatenate([np.array([P_ghost_start]), P, np.array([P_ghost_end])])

    y1_ghost_start = func.quadratic_extrapolation(column_grid["xWalls"][column_grid["nGhost"]], y1_inlet, column_grid["xCentres"][column_grid["nGhost"]], y1[0], column_grid["xCentres"][column_grid["nGhost"]+1], y1[1], column_grid["xCentres"][0])
    y1_ghost_end = func.quadratic_extrapolation(column_grid["xWalls"][-(column_grid["nGhost"]+1)], y1_outlet, column_grid["xCentres"][-(column_grid["nGhost"]+1)], y1[-1], column_grid["xCentres"][-(column_grid["nGhost"]+2)], y1[-2], column_grid["xCentres"][-1])
    y1_all = np.concatenate((np.array([y1_ghost_start]), y1,np.array([y1_ghost_end]))) 

    y2_ghost_start = func.quadratic_extrapolation(column_grid["xWalls"][column_grid["nGhost"]], y2_inlet, column_grid["xCentres"][column_grid["nGhost"]], y2[0], column_grid["xCentres"][column_grid["nGhost"]+1], y2[1], column_grid["xCentres"][0])
    y2_ghost_end = func.quadratic_extrapolation(column_grid["xWalls"][-(column_grid["nGhost"]+1)], y2_outlet, column_grid["xCentres"][-(column_grid["nGhost"]+1)], y2[-1], column_grid["xCentres"][-(column_grid["nGhost"]+2)], y2[-2], column_grid["xCentres"][-1])
    y2_all = np.concatenate((np.array([y2_ghost_start]), y2,np.array([y2_ghost_end]))) 

    y3_ghost_start = func.quadratic_extrapolation(column_grid["xWalls"][column_grid["nGhost"]], y3_inlet, column_grid["xCentres"][column_grid["nGhost"]], y3[0], column_grid["xCentres"][column_grid["nGhost"]+1], y3[1], column_grid["xCentres"][0])
    y3_ghost_end = func.quadratic_extrapolation(column_grid["xWalls"][-(column_grid["nGhost"]+1)], y3_outlet, column_grid["xCentres"][-(column_grid["nGhost"]+1)], y3[-1], column_grid["xCentres"][-(column_grid["nGhost"]+2)], y3[-2], column_grid["xCentres"][-1])
    y3_all = np.concatenate((np.array([y3_ghost_start]), y3,np.array([y3_ghost_end]))) 
    
    T_ghost_start = func.quadratic_extrapolation(column_grid["xWalls"][column_grid["nGhost"]], T_inlet, column_grid["xCentres"][column_grid["nGhost"]], T[0], column_grid["xCentres"][column_grid["nGhost"]+1], T[1], column_grid["xCentres"][0])
    T_ghost_end = func.quadratic_extrapolation(column_grid["xWalls"][-(column_grid["nGhost"]+1)], T_outlet, column_grid["xCentres"][-(column_grid["nGhost"]+1)], T[-1], column_grid["xCentres"][-(column_grid["nGhost"]+2)], T[-2], column_grid["xCentres"][-1])
    T_all = np.concatenate((np.array([T_ghost_start]), T,np.array([T_ghost_end])))

    Tw_ghost_start = Tw[0] + (column_grid["xCentres"][0] - column_grid["xCentres"][1]) * (Tw[0] - Tw_inlet) / (column_grid["xCentres"][column_grid["nGhost"]] - column_grid["xWalls"][column_grid["nGhost"]])
    Tw_ghost_end = Tw[-1] - (column_grid["xCentres"][-1] - column_grid["xCentres"][-column_grid["nGhost"]-1]) * (Tw[-1] - Tw_outlet) / (column_grid["xCentres"][-(column_grid["nGhost"]+1)] - column_grid["xWalls"][-(column_grid["nGhost"]+1)]) 
    Tw_all = np.concatenate([np.array([Tw_ghost_start]), Tw, np.array([Tw_ghost_end])])

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
    mu = func.calculate_gas_viscosity()
    a = 1.75 * (1- bed_properties["bed_voidage"]) * avg_density_walls / (bed_properties["bed_voidage"]**3 * bed_properties["particle_diameter"])
    b = 12 * mu * (1-bed_properties["bed_voidage"])**2 / (bed_properties["bed_voidage"]**3 * bed_properties["particle_diameter"]**2)
    c = np.abs(dPdz_walls[:])
    dominant = b**2+4*a*c
        
    if np.any(dominant < 0):
        raise ValueError("Negative value under square root in velocity calculation. Check your inputs and boundary conditions.")
    v_walls = np.array((-b + np.sqrt(dominant)) / (2*a)) # should have N + 1 values 
    v_walls = np.multiply(-np.sign(dPdz_walls),v_walls)  # make sure velocity is in the correct direction
    v_walls[0] = v_inlet  # set inlet velocity
    v_walls[-1] = v_outlet # set outlet velocity

    
    return P_walls, T_walls, Tw_walls, y1_walls, y2_walls, y3_walls, v_walls, dTdz_walls, dTwdz_walls

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
    P_walls, T_walls, Tw_walls, y1_walls, y2_walls, y3_walls, v_walls, dTdz_walls, dTwdz_walls = calculate_internal_wall_values(
        P_all, T_all, Tw_all, y1_all, y2_all, y3_all,
        P_inlet, P_outlet, T_inlet, T_outlet, Tw_inlet, Tw_outlet,
        y1_inlet, y1_outlet, y2_inlet, y2_outlet,
        y3_inlet, y3_outlet, v_inlet, v_outlet,
        dPdz_inlet, dPdz_outlet, dTwdz_inlet, dTwdz_outlet,
        bed_properties, column_grid)
    
    
    
    """   
    Calculate the derivatives of the state variables.
    This section computes the mass transfer and energy balance equations for the column.
    """
    # Solid phase balance for adsorbed components
    k1 = 0.1 # s-1 linear driving force mass transfer constant
    dn1dt = k1 * (func.adsorption_isotherm_1(P, T, y1, y2)[0]-n1)*bed_properties["rho_bed"]/(1-bed_properties["bed_voidage"]) 
    # \frac{\partial q_1}{\partial t} = k_1 (q_1^* - q_1)
    deltaH_1 = func.adsorption_isotherm_1(P, T, y1, y2)[1] * 1000 # Heat of adsorption for component 1 (J/mol)

    k2 = 0.1 # s-1 linear driving force mass transfer constant
    dn2dt = k2*(func.adsorption_isotherm_2(P, T, y2)[0]-n2)*bed_properties["rho_bed"]/(1-bed_properties["bed_voidage"])
    # \frac{\partial q_2}{\partial t} = k_2 (q_2^* - q_2)
    deltaH_2 = func.adsorption_isotherm_2(P, T, y2)[1] * 1000  # Heat of adsorption for component 2 (J/mol)

    # Heat transfer
    Cp_g = func.calculate_gas_heat_capacity()
    Cp_solid = bed_properties["solid_heat_capacity"]
    Cp_1 = bed_properties["heat_capacity_1"]
    Cp_2 = bed_properties["heat_capacity_2"]
    K_z = func.calculate_gas_thermal_conductivity()
    K_wall = func.calculate_wall_thermal_conductivity()
    h_bed = func.heat_transfer_coefficient()[0]
    h_wall = func.heat_transfer_coefficient()[1]
    D_l = func.calculate_axial_dispersion_coefficient(bed_properties, inlet_values)

    #Column energy balance
    q1 = n1 * bed_properties["rho_bed"]/(1-bed_properties["bed_voidage"])
    q2 = n2 * bed_properties["rho_bed"]/(1-bed_properties["bed_voidage"])
    a_2 = (1 - bed_properties["bed_voidage"]) / bed_properties["bed_voidage"] * (Cp_solid * bed_properties["rho_bed"] + Cp_1 * q1 + Cp_2 * q2) + Cp_g * P/(bed_properties["R"] * T)
    dTdt = 1/a_2 * (
        K_z/bed_properties["bed_voidage"] * (1/(column_grid["deltaZ"][1:-1]) * (dTdz_walls[1:num_cells+1] - dTdz_walls[:num_cells])) 
        - Cp_g/(bed_properties["R"] * column_grid["deltaZ"][1:-1]) * (v_walls[1:num_cells+1] * P_walls[1:num_cells+1] - v_walls[:num_cells] * P_walls[:num_cells]) -
        Cp_g/bed_properties["R"]*(-(1-bed_properties["bed_voidage"])/bed_properties["bed_voidage"] *bed_properties["R"] * T * (dn1dt + dn2dt) 
        - T/(column_grid["deltaZ"][1:-1]*bed_properties["bed_voidage"]) * (P_walls[1:num_cells+1] * v_walls[1:num_cells+1]/T_walls[1:num_cells+1] - P_walls[:num_cells] * v_walls[:num_cells]/T_walls[:num_cells]))
        + (1 - bed_properties["bed_voidage"]) / bed_properties["bed_voidage"] * (np.abs(deltaH_1) * dn1dt + np.abs(deltaH_2) * dn2dt)
        + (1 - bed_properties["bed_voidage"]) / bed_properties["bed_voidage"] * T * (Cp_1 * dn1dt + Cp_2 * dn2dt)
        + 2 * h_bed * (T - Tw) / (bed_properties["bed_voidage"] * bed_properties["inner_bed_diameter"]) )
    # \frac{\partial T}{\partial t} = \frac{1}{a_2}\left[
    # \frac{K_z}{\varepsilon \Delta z}\left(\frac{\partial T}{\partial z}|_{i+1/2} - \frac{\partial T}{\partial z}|_{i-1/2}\right) - \frac{C_{p,g}}{R \Delta z} \left(P_{i+1/2} \ v_{i+1/2} - P_{i-1/2}v_{i-1/2} \right) \right.\\
    # - \frac{C_{p,g}}{R} \left( - \dfrac{(1-\varepsilon)}{\varepsilon}{RT} \left(\frac{\partial q_1}{\partial t}+\frac{\partial q_2}{\partial t}\right) - \frac{1}{\varepsilon} \frac{T}{\Delta z} \left( \frac{P_{i+1/2} \ v_{i+1/2}}{T_{i+1/2}}-\frac{P_{i-1/2} \ v_{i-1/2}}{T_{i-1/2}} \right) \right) \\ \left. + \frac{1-\varepsilon}{\varepsilon} \left(  \Delta H_1 \frac{\partial q_1}{\partial t} +\Delta H_2 \frac{\partial q_2}{\partial t}\right)
    # + \frac{1-\varepsilon}{\varepsilon}  C_{p,ads} T \left( \frac{\partial q_1}{\partial t} + \frac{\partial q_2}{\partial t}\right) + \frac{2h_{bed}}{R_{in}}(T-T_w)\right]"""
    
    
    # Total mass balance
    dPdt = ( P/T * dTdt - (1 - bed_properties["bed_voidage"]) / bed_properties["bed_voidage"] * bed_properties["R"] * T * (dn1dt + dn2dt) - 
            T / bed_properties["bed_voidage"] * 1 / column_grid["deltaZ"][1:-1] * 
            (P_walls[1:num_cells+1] * v_walls[1:num_cells+1] / T_walls[1:num_cells+1] - P_walls[:num_cells] * v_walls[:num_cells] / T_walls[:num_cells])    )
    # \dfrac{\partial P}{\partial t} = \dfrac{P}{T}\dfrac{\partial T}{\partial t} - \dfrac{(1-\varepsilon)}{\varepsilon}{RT} \left(\frac{\partial q_1}{\partial t}+\frac{\partial q_2}{\partial t}\right)
    #- \frac{1}{\varepsilon} \frac{T}{\Delta z} \left( \frac{P_{i+1/2} \ v_{i+1/2}}{T_{i+1/2}}-\frac{P_{i-1/2} \ v_{i-1/2}}{T_{i-1/2}} \right)"""

    # Wall energy balance
    d2Twdt2 = 1/(column_grid["deltaZ"][1:-1]) *(dTwdz_walls[1:num_cells+1] - dTwdz_walls[:num_cells])

    dTwdt = 1 / (bed_properties["wall_heat_capacity"] * bed_properties["wall_density"]) * (
        K_wall * d2Twdt2 - 2 * bed_properties["inner_bed_diameter"] * h_bed * (T - Tw) /
        (bed_properties["outer_bed_diameter"]**2 - bed_properties["inner_bed_diameter"]**2) -
        2 * bed_properties["outer_bed_diameter"] * h_wall * (Tw - inlet_values["feed_temperature"]) /
        (bed_properties["outer_bed_diameter"]**2 - bed_properties["inner_bed_diameter"]**2) )
    #"""\frac{\partial T_w}{\partial t} =
    #\frac{1}{\rho_w \ C_{p,wall}} \left(\frac{K_{wall}}{\Delta z}\left(\frac{\partial T_w}{\partial z}|_{i+1/2} - \frac{\partial T_w}{\partial z}|_{i-1/2}\right) 
    #+ \frac{2 r_{in} \ h_{bed}}{r_{out}^2-r_{in}^2} \left(T-T_w \right) - \frac{2 r_{out} \ h_{wall}}{r_{out}^2-r_{in}^2} \left(T_w-T_a \right)\right)"""
    

    # Component mass balances
    dy1dt = (-y1/P*dPdt + y1/T*dTdt - 
            (1-bed_properties["bed_voidage"])/bed_properties["bed_voidage"]*bed_properties["R"]*T/P*dn1dt - 
            1/bed_properties["bed_voidage"]*T/P*1/column_grid["deltaZ"][1:-1]*
            (P_walls[1:num_cells+1]*v_walls[1:num_cells+1]*y1_walls[1:num_cells+1]/T_walls[1:num_cells+1] - 
             P_walls[:num_cells]*v_walls[:num_cells]*y1_walls[:num_cells]/T_walls[:num_cells]) + 
            D_l * T/P * 1/column_grid["deltaZ"][1:-1] * 
            (P_walls[1:num_cells+1]/T_walls[1:num_cells+1]*(y1_all[2:num_cells+2]-y1_all[1:num_cells+1])/column_grid["deltaZ"][1:num_cells+1] - 
             P_walls[0:num_cells]/T_walls[0:num_cells]*(y1_all[1:num_cells+1]-y1_all[:num_cells])/column_grid["deltaZ"][0:num_cells]))
    #""" \dfrac{\partial y_1}{\partial t} = \dfrac{y_1}{P}\dfrac{\partial P}{\partial t} + \dfrac{y_1}{T}\dfrac{\partial T}{\partial t} 
    #- \dfrac{(1-\varepsilon)}{\varepsilon}\frac{RT}{P}\frac{\partial q_1}{\partial t} - \frac{1}{\varepsilon}\frac{T}{P}\frac{1}{\Delta z} 
    #\left( \frac{P_{i+1/2} \ y_{1,i+1/2} \ v_{i+1/2}}{T_{i+1/2}}-\frac{P_{i-1/2} \ y_{1,i-1/2} \ v_{i-1/2}}{T_{i-1/2}} \right) \\ 
    #+ D_l \  \frac{T}{P} \ \frac{1}{\Delta z} \left( \frac{P_{i+1/2}}{T_{i+1/2}} \ \frac{y_{1,i+1} - y_{1,i}}{\Delta z} - \frac{P_{i-1/2}}{T_{i-1/2}} \ \frac{y_{1,i} - y_{1,i-1}}{\Delta z}\right)"""

    dy2dt = (-y2/P*dPdt + y2/T*dTdt - 
            (1-bed_properties["bed_voidage"])/bed_properties["bed_voidage"]*bed_properties["R"]*T/P*dn2dt - 
            1/bed_properties["bed_voidage"]*T/P*1/column_grid["deltaZ"][1:-1]*
            (P_walls[1:num_cells+1]*v_walls[1:num_cells+1]*y2_walls[1:num_cells+1]/T_walls[1:num_cells+1] - 
             P_walls[:num_cells]*v_walls[:num_cells]*y2_walls[:num_cells]/T_walls[:num_cells]) + 
            D_l * T/P * 1/column_grid["deltaZ"][1:-1] * 
            (P_walls[1:num_cells+1]/T_walls[1:num_cells+1]*(y2_all[2:num_cells+2]-y2_all[1:num_cells+1])/column_grid["deltaZ"][1:num_cells+1] - 
             P_walls[0:num_cells]/T_walls[0:num_cells]*(y2_all[1:num_cells+1]-y2_all[:num_cells])/column_grid["deltaZ"][0:num_cells]))
    #"""\dfrac{\partial y_2}{\partial t} = \dfrac{y_2}{P}\dfrac{\partial P}{\partial t} + \dfrac{y_2}{T}\dfrac{\partial T}{\partial t} - \dfrac{(1-\varepsilon)}{\varepsilon}\frac{RT}{P}\frac{\partial q_2}{\partial t} 
    #- \frac{1}{\varepsilon}\frac{T}{P}\frac{1}{\Delta z} \left( \frac{P_{i+1/2} \ y_{2,i+1/2} \ v_{i+1/2}}{T_{i+1/2}}-\frac{P_{i-1/2} \ y_{2,i-1/2} \ v_{i-1/2}}{T_{i-1/2}} \right)
    #\\  + D_l \  \frac{T}{P} \ \frac{1}{\Delta z} \left( \frac{P_{i+1/2}}{T_{i+1/2}} \ \frac{y_{2,i+1} - y_{2,i}}{\Delta z} - \frac{P_{i-1/2}}{T_{i-1/2}} \ \frac{y_{2,i} - y_{2,i-1}}{\Delta z}\right)"""
             
    dy3dt = (-y3/P*dPdt + y3/T*dTdt - 
            1/bed_properties["bed_voidage"]*T/P*1/column_grid["deltaZ"][1:-1]*
            (P_walls[1:num_cells+1]*v_walls[1:num_cells+1]*y3_walls[1:num_cells+1]/T_walls[1:num_cells+1] - 
             P_walls[:num_cells]*v_walls[:num_cells]*y3_walls[:num_cells]/T_walls[:num_cells]) + 
            D_l * T/P * 1/column_grid["deltaZ"][1:-1] * 
            (P_walls[1:num_cells+1]/T_walls[1:num_cells+1]*(y3_all[2:num_cells+2]-y3_all[1:num_cells+1])/column_grid["deltaZ"][1:num_cells+1] - 
             P_walls[0:num_cells]/T_walls[0:num_cells]*(y3_all[1:num_cells+1]-y3_all[:num_cells])/column_grid["deltaZ"][0:num_cells]))
    #""" \dfrac{\partial y_3}{\partial t} = \dfrac{y_3}{P}\dfrac{\partial P}{\partial t} + \dfrac{y_3}{T}\dfrac{\partial T}{\partial t} 
    #- \frac{1}{\varepsilon}\frac{T}{P}\frac{1}{\Delta z} \left( \frac{P_{i+1/2} \ y_{3,i+1/2} \ v_{i+1/2}}{T_{i+1/2}}-\frac{P_{i-1/2} \ y_{3,i-1/2} \ v_{i-1/2}}{T_{i-1/2}} \right)
    #\\ + D_l \  \frac{T}{P} \ \frac{1}{\Delta z} \left( \frac{P_{i+1/2}}{T_{i+1/2}} \ \frac{y_{3,i+1} - y_{3,i}}{\Delta z} - \frac{P_{i-1/2}}{T_{i-1/2}} \ \frac{y_{3,i} - y_{3,i-1}}{\Delta z}\right)"""


    
    # Inlet and outlet calculations for mass balance error
    dF1dt = bed_properties["bed_voidage"] * bed_properties["column_area"] / (bed_properties["R"] * T_walls[0]) * v_walls[0]*P_walls[0] * y1_walls[0]
    dF2dt = bed_properties["bed_voidage"] * bed_properties["column_area"] / (bed_properties["R"] * T_walls[0]) * v_walls[0]*P_walls[0] * y2_walls[0]
    dF3dt = bed_properties["bed_voidage"] * bed_properties["column_area"] / (bed_properties["R"] * T_walls[0]) * v_walls[0]*P_walls[0] * y3_walls[0]
    dF4dt = bed_properties["bed_voidage"] * bed_properties["column_area"] / (bed_properties["R"] * T_walls[0]) * v_walls[0]*P_walls[0] * (1 - y1_walls[0]-y2_walls[0]-y3_walls[0])
    dF5dt = bed_properties["bed_voidage"] * bed_properties["column_area"] / (bed_properties["R"] * T_walls[-1]) * v_walls[-1]*P_walls[-1] * y1_walls[-1]
    dF6dt = bed_properties["bed_voidage"] * bed_properties["column_area"] / (bed_properties["R"] * T_walls[-1]) * v_walls[-1]*P_walls[-1] * y2_walls[-1]
    dF7dt = bed_properties["bed_voidage"] * bed_properties["column_area"] / (bed_properties["R"] * T_walls[-1]) * v_walls[-1]*P_walls[-1] * y3_walls[-1]
    dF8dt = bed_properties["bed_voidage"] * bed_properties["column_area"] / (bed_properties["R"] * T_walls[-1]) * v_walls[-1]*P_walls[-1] * (1 - y1_walls[-1]-y2_walls[-1]-y3_walls[-1])
    dFdt = np.array([dF1dt, dF2dt, dF3dt, dF4dt, dF5dt, dF6dt, dF7dt, dF8dt])

    #Inlet and outlet calculations for energy balance error
    dE1dt = (bed_properties["bed_voidage"] * bed_properties["column_area"] * Cp_g * v_walls[0] * T_walls[0] * P_walls[0] / (bed_properties["R"] * T_walls[0]))
    dE2dt = (bed_properties["bed_voidage"] * bed_properties["column_area"] * Cp_g * v_walls[-1] * T_walls[-1] * P_walls[-1] / (bed_properties["R"] * T_walls[-1]))
    dEdt = np.array([dE1dt, dE2dt])

    # Combine derivatives into a single vector
    derivatives = np.concatenate([dPdt, dTdt, dTwdt, dy1dt, dy2dt, dy3dt, dn1dt, dn2dt, dFdt, dEdt]) 
    
    return derivatives