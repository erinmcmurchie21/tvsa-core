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
import additional_functions_haghpanagh as func
import scipy.integrate

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

### Split results vector from previous time step #######

def data_prep(results_vector, num_cells):
    """
    Split the results vector into individual variables.
    Four component system where y1 + y2 + y3 + y4 = 1; two adsorbing components with (CO2 and H2O)
    Vector contains P, T, Tw, y1, y2, y3, n1, n2, F
Returns:
    --------
    tuple : (P, T, Tw, y1, y2, y3, n1, n2, F, E)
        P : Pressure [Pa]
        T : Gas temperature [K] 
        Tw : Wall temperature [K]
        y1 : Mole fraction of CO2 [-]
        y2 : Mole fraction of H2O [-]
        y3 : Mole fraction of N2 [-]
        n1 : Adsorbed concentration of CO2 [mol/m³]
        n2 : Adsorbed concentration of H2O [mol/m³]
        F : Flow rate tracking variables [mol/s]
        E : Energy flow tracking variables [J/s]
    """
    # Extract state variables from the combined vector
    P = results_vector[:num_cells]                              # Pressure
    T = results_vector[num_cells:2*num_cells]                   # Gas temperature
    Tw = results_vector[2*num_cells:3*num_cells]                # Wall temperature
    y1 = results_vector[3*num_cells:4*num_cells]                # CO2 mole fraction
    y2 = results_vector[4*num_cells:5*num_cells]                # H2O mole fraction
    y3 = results_vector[5*num_cells:6*num_cells]                # N2 mole fraction
    n1 = results_vector[6*num_cells:7*num_cells]                # CO2 adsorbed concentration
    n2 = results_vector[7*num_cells:8*num_cells]                # H2O adsorbed concentration
    F = results_vector[8*num_cells:8*num_cells+8]               # Flow tracking (8 components)
    E = results_vector[8*num_cells+8:]                          # Energy tracking (2 components)

    return P, T, Tw, y1, y2, y3, n1, n2, F, E
    
def inlet_boundary_conditions(P, T, Tw, y1, y2, y3, column_grid, bed_properties, inlet_values):

    """
    Apply inlet boundary conditions at z=0.
    
    Supports two inlet types:
    - "mass_flow": Specified mass flow rate with convective boundary conditions
    - "closed": Closed inlet with zero-gradient boundary conditions
    
    Parameters:
    -----------
    P, T, Tw, y1, y2, y3 : numpy.ndarray
        Current state variables at cell centers
    column_grid : dict
        Grid parameters including cell positions and sizes
    bed_properties : dict
        Physical properties of the bed and column
    inlet_values : dict
        Inlet operating conditions and type
        
    Returns:
    --------
    tuple : Boundary values and derivatives at inlet
        (P_inlet, T_inlet, Tw_inlet, y1_inlet, y2_inlet, y3_inlet, 
         v_inlet, dPdz_inlet, dTwdz_inlet)
    """
    
    if inlet_values["inlet_type"] == "mass_flow":
        # Calculate transport properties at inlet conditions
        rho_gas_inlet = func.calculate_gas_density(P[0], T[0])  # [mol/m³]
        mu = func.calculate_gas_viscosity()                     # [Pa·s]
        D_l = func.calculate_axial_dispersion_coefficient(bed_properties, inlet_values)  # [m²/s]
        v_inlet = inlet_values["velocity"]                      # [m/s]
        Cp_g = func.calculate_gas_heat_capacity()               # [J/mol·K]
        thermal_diffusivity = func.calculate_gas_thermal_conductivity() / (Cp_g * rho_gas_inlet)  # [m²/s]

        # Calculate inlet mole fractions using convective boundary conditions
        # Boundary condition: dy/dz = -(v/D_l)(y_feed - y_inlet)
        #y1_inlet = func.quadratic_extrapolation_derivative_nonzero(
        #    column_grid["xCentres"][column_grid["nGhost"]], y1[0], 
        #    column_grid["xCentres"][column_grid["nGhost"]+1], y1[1],
        #    column_grid["xWalls"][column_grid["nGhost"]], 
        #    -np.abs(v_inlet / D_l), inlet_values["y1_feed_value"])

        #y2_inlet = func.quadratic_extrapolation_derivative_nonzero(
        #    column_grid["xCentres"][column_grid["nGhost"]], y2[0], 
        #   column_grid["xCentres"][column_grid["nGhost"]+1], y2[1],
        #   column_grid["xWalls"][column_grid["nGhost"]], 
        #   -np.abs(v_inlet / D_l), inlet_values["y2_feed_value"])


        #y3_inlet = func.quadratic_extrapolation_derivative_nonzero(
        #   column_grid["xCentres"][column_grid["nGhost"]], y3[0], 
        #   column_grid["xCentres"][column_grid["nGhost"]+1], y3[1],
        #   column_grid["xWalls"][column_grid["nGhost"]], 
        #   -np.abs(v_inlet / D_l), inlet_values["y3_feed_value"])

        y1_inlet = (y1[0] + v_inlet/D_l * inlet_values["y1_feed_value"] * column_grid["xCentres"][column_grid["nGhost"]]) / (1 + v_inlet/D_l * column_grid["xCentres"][column_grid["nGhost"]])
        y2_inlet = (y2[0] + v_inlet/D_l * inlet_values["y2_feed_value"] * column_grid["xCentres"][column_grid["nGhost"]]) / (1 + v_inlet/D_l * column_grid["xCentres"][column_grid["nGhost"]])
        y3_inlet = (y3[0] + v_inlet/D_l * inlet_values["y3_feed_value"] * column_grid["xCentres"][column_grid["nGhost"]]) / (1 + v_inlet/D_l * column_grid["xCentres"][column_grid["nGhost"]])

        # Calculate average gas density for Ergun equation
        avg_density_inlet = rho_gas_inlet / 1000 * (
            bed_properties["MW_1"] * y1_inlet + 
            bed_properties["MW_2"] * y2_inlet + 
            bed_properties["MW_3"] * y3_inlet + 
            bed_properties["MW_4"] * (1 - y1_inlet - y2_inlet - y3_inlet))  # [kg/m³]

        # Calculate pressure drop using Ergun equation
        # dP/dz = -(150μ(1-ε)²v)/(ε³dp²) - (1.75ρ(1-ε)v²)/(ε³dp)
        dPdz_inlet = (
            1.75 * (1 - bed_properties["bed_voidage"]) * avg_density_inlet * v_inlet**2 / 
            (bed_properties["bed_voidage"]**3 * bed_properties["particle_diameter"]) + 
            150 * mu * (1 - bed_properties["bed_voidage"])**2 * v_inlet / 
            (bed_properties["bed_voidage"]**3 * bed_properties["particle_diameter"]**2))
        
        # Calculate inlet pressure from pressure gradient
        P_inlet = P[0] - dPdz_inlet * (
            column_grid["xWalls"][int(column_grid["nGhost"])] - 
            column_grid["xCentres"][int(column_grid["nGhost"])])
    
        # Calculate inlet temperature using convective boundary condition
        # dT/dz = -(v·Pe_h)(T_feed - T_inlet)
        Pe_h = bed_properties["bed_voidage"] / thermal_diffusivity  # Péclet number
        T_inlet = (T[0] + column_grid["xCentres"][column_grid["nGhost"]] * v_inlet * Pe_h * inlet_values["feed_temperature"])/(1 + column_grid["xCentres"][column_grid["nGhost"]] * v_inlet * Pe_h)

        # Wall temperature boundary conditions
        Tw_inlet = Tw[0]
        dTwdz_inlet = 0

    elif inlet_values["inlet_type"] == "closed":
        # Closed inlet: zero gradients for all variables
        dPdz_inlet = 0
        P_inlet = P[0]
        v_inlet = 0
        
        # Use zero-gradient extrapolation for composition variables
        y1_inlet = func.quadratic_extrapolation_derivative(
            column_grid["xCentres"][column_grid["nGhost"]], y1[0], 
            column_grid["xCentres"][column_grid["nGhost"]+1], y1[1],
            column_grid["xWalls"][column_grid["nGhost"]])
        
        y2_inlet = func.quadratic_extrapolation_derivative(
            column_grid["xCentres"][column_grid["nGhost"]], y2[0], 
            column_grid["xCentres"][column_grid["nGhost"]+1], y2[1],
            column_grid["xWalls"][column_grid["nGhost"]])
        
        y3_inlet = func.quadratic_extrapolation_derivative(
            column_grid["xCentres"][column_grid["nGhost"]], y3[0], 
            column_grid["xCentres"][column_grid["nGhost"]+1], y3[1],
            column_grid["xWalls"][column_grid["nGhost"]])
        
        T_inlet = func.quadratic_extrapolation_derivative(
            column_grid["xCentres"][column_grid["nGhost"]], T[0], 
            column_grid["xCentres"][column_grid["nGhost"]+1], T[1],
            column_grid["xWalls"][column_grid["nGhost"]])
        
        Tw_inlet = Tw[0]                            
        dTwdz_inlet = 0
    
    return P_inlet, T_inlet, Tw_inlet, y1_inlet, y2_inlet, y3_inlet, v_inlet, dPdz_inlet, dTwdz_inlet

def outlet_boundary_conditions(P, T, Tw, y1, y2, y3, column_grid, bed_properties, outlet_values):
    """
    Apply outlet boundary conditions at z=L.
    
    Supports three outlet types:
    - "pressure": Fixed pressure with extrapolated composition
    - "closed": Closed outlet with zero-gradient boundary conditions  
    - "mass_flow": Fixed mass flow rate (implementation incomplete)
    
    Parameters:
    -----------
    P, T, Tw, y1, y2, y3 : numpy.ndarray
        Current state variables at cell centers
    column_grid : dict
        Grid parameters
    bed_properties : dict
        Physical properties
    outlet_values : dict
        Outlet operating conditions and type
        
    Returns:
    --------
    tuple : Boundary values and derivatives at outlet
        (P_outlet, T_outlet, Tw_outlet, y1_outlet, y2_outlet, y3_outlet,
         v_outlet, dPdz_outlet, dTwdz_outlet)
    """

    if outlet_values["outlet_type"] == "pressure":
        # Fixed outlet pressure
        P_outlet = outlet_values["outlet_pressure"]
        
        # Extrapolate composition variables using quadratic interpolation
        
        #y1_outlet = func.quadratic_extrapolation_derivative(
            #column_grid["xCentres"][-(column_grid["nGhost"]+1)], y1[-1], 
            #column_grid["xCentres"][-(column_grid["nGhost"]+2)], y1[-2],
            #column_grid["xWalls"][-(column_grid["nGhost"]+1)])
        
        #y2_outlet = func.quadratic_extrapolation_derivative(
            #column_grid["xCentres"][-(column_grid["nGhost"]+1)], y2[-1], 
            #column_grid["xCentres"][-(column_grid["nGhost"]+2)], y2[-2],
            #column_grid["xWalls"][-(column_grid["nGhost"]+1)])
        
        #y3_outlet = func.quadratic_extrapolation_derivative(
            #column_grid["xCentres"][-(column_grid["nGhost"]+1)], y3[-1], 
            #column_grid["xCentres"][-(column_grid["nGhost"]+2)], y3[-2],
            #column_grid["xWalls"][-(column_grid["nGhost"]+1)])

        y1_outlet = y1[-1]
        y2_outlet = y2[-1]
        y3_outlet = y3[-1]
        
        # Zero gradient for temperature
        #T_outlet = func.quadratic_extrapolation_derivative(
        #    column_grid["xCentres"][-(column_grid["nGhost"]+1)], T[-1], 
        #    column_grid["xCentres"][-(column_grid["nGhost"]+2)], T[-2],
        #    column_grid["xWalls"][-(column_grid["nGhost"]+1)])

        T_outlet = T[-1]
        
        Tw_outlet = Tw[-1]
        
        # Calculate outlet velocity from Ergun equation
        mu = func.calculate_gas_viscosity()
        rho_gas_outlet = P_outlet / bed_properties["R"] / T_outlet  # [mol/m³]
        
        # Convert to mass density [kg/m³]
        avg_density_outlet = rho_gas_outlet / 1000 * (
            bed_properties["MW_1"] * y1_outlet + 
            bed_properties["MW_2"] * y2_outlet + 
            bed_properties["MW_3"] * y3_outlet + 
            bed_properties["MW_4"] * (1 - y1_outlet - y2_outlet - y3_outlet))
        
        # Calculate pressure gradient
        dPdz_outlet = (P_outlet - P[-1]) / (
            column_grid["xWalls"][-1] - column_grid["xCentres"][-1])
        
        # Solve Ergun equation: av² + bv + c = 0
        a = 1.75 * (1 - bed_properties["bed_voidage"]) * avg_density_outlet / (
            bed_properties["bed_voidage"]**3 * bed_properties["particle_diameter"])
        b = 150 * mu * (1 - bed_properties["bed_voidage"])**2 / (
            bed_properties["bed_voidage"]**3 * bed_properties["particle_diameter"]**2)
        c = np.abs(dPdz_outlet)
        
        v_outlet = -np.sign(dPdz_outlet) * ((-b + np.sqrt(b**2 + 4*a*c)) / (2*a))
        dTwdz_outlet = 0

    elif outlet_values["outlet_type"] == "closed":
        # Closed outlet: zero gradients
        P_outlet = P[-(column_grid["nGhost"]+1)]
        
        y1_outlet = func.quadratic_extrapolation_derivative(
            column_grid["xCentres"][-(column_grid["nGhost"]+1)], y1[-1], 
            column_grid["xCentres"][-(column_grid["nGhost"]+2)], y1[-2], 
            column_grid["xWalls"][-(column_grid["nGhost"]+1)])
        
        y2_outlet = func.quadratic_extrapolation_derivative(
            column_grid["xCentres"][-(column_grid["nGhost"]+1)], y2[-1], 
            column_grid["xCentres"][-(column_grid["nGhost"]+2)], y2[-2],
            column_grid["xWalls"][-(column_grid["nGhost"]+1)])
        
        y3_outlet = func.quadratic_extrapolation_derivative(
            column_grid["xCentres"][-(column_grid["nGhost"]+1)], y3[-1], 
            column_grid["xCentres"][-(column_grid["nGhost"]+2)], y3[-2],
            column_grid["xWalls"][-(column_grid["nGhost"]+1)])
        
        T_outlet = func.quadratic_extrapolation_derivative(
            column_grid["xCentres"][-(column_grid["nGhost"]+1)], T[-1], 
            column_grid["xCentres"][-(column_grid["nGhost"]+2)], T[-2],
            column_grid["xWalls"][-(column_grid["nGhost"]+1)])
        
        Tw_outlet = Tw[-1]
        v_outlet = 0
        dPdz_outlet = 0
        dTwdz_outlet = 0

    elif outlet_values["outlet_type"] == "mass_flow":
        # Mass flow outlet (implementation incomplete)
        P_outlet = P[-1]
        # TODO: Implement mass flow boundary condition
        # Need to specify other variables based on mass flow constraint

    return P_outlet, T_outlet, Tw_outlet, y1_outlet, y2_outlet, y3_outlet, v_outlet, dPdz_outlet, dTwdz_outlet

def ghost_cell_calculations(P, T, Tw, y1, y2, y3, 
                            P_inlet, P_outlet, T_inlet, T_outlet, Tw_inlet, Tw_outlet, 
                            y1_inlet, y1_outlet, y2_inlet, y2_outlet, 
                            y3_inlet, y3_outlet, column_grid):
    """
    Calculate ghost cell values using boundary conditions.
    
    Ghost cells are fictitious cells outside the computational domain used to
    implement boundary conditions in finite volume methods.
    
    Parameters:
    -----------
    P, T, Tw, y1, y2, y3 : numpy.ndarray
        State variables at interior cell centers
    *_inlet, *_outlet : float
        Boundary values at inlet and outlet
    column_grid : dict
        Grid parameters
        
    Returns:
    --------
    tuple : Extended arrays including ghost cells
        (P_all, T_all, Tw_all, y1_all, y2_all, y3_all)
    """
    
    # Pressure ghost cells using linear extrapolation
    P_ghost_start = P[0] + (column_grid["xCentres"][0] - column_grid["xCentres"][1]) * \
                    (P[0] - P_inlet) / (column_grid["xCentres"][column_grid["nGhost"]] - 
                                       column_grid["xWalls"][column_grid["nGhost"]])
    
    P_ghost_end = P[-1] - (column_grid["xCentres"][-1] - column_grid["xCentres"][-column_grid["nGhost"]-1]) * \
                  (P[-1] - P_outlet) / (column_grid["xCentres"][-(column_grid["nGhost"]+1)] - 
                                       column_grid["xWalls"][-(column_grid["nGhost"]+1)])
    
    P_all = np.concatenate([np.array([P_ghost_start]), P, np.array([P_ghost_end])])

    # Composition ghost cells using quadratic extrapolation
    y1_ghost_start = y1[0] + (column_grid["xCentres"][0] - column_grid["xCentres"][1]) * \
                    (y1[0] - y1_inlet) / (column_grid["xCentres"][column_grid["nGhost"]] - 
                                       column_grid["xWalls"][column_grid["nGhost"]])
    
    y1_ghost_end = y1[-1] - (column_grid["xCentres"][-1] - column_grid["xCentres"][-column_grid["nGhost"]-1]) * \
                  (y1[-1] - y1_outlet) / (column_grid["xCentres"][-(column_grid["nGhost"]+1)] - 
                                       column_grid["xWalls"][-(column_grid["nGhost"]+1)])

    """y1_ghost_start = func.quadratic_extrapolation(
        column_grid["xWalls"][column_grid["nGhost"]], y1_inlet, 
        column_grid["xCentres"][column_grid["nGhost"]], y1[0], 
        column_grid["xCentres"][column_grid["nGhost"]+1], y1[1], 
        column_grid["xCentres"][0])
    
    y1_ghost_end = func.quadratic_extrapolation(
        column_grid["xWalls"][-(column_grid["nGhost"]+1)], y1_outlet, 
        column_grid["xCentres"][-(column_grid["nGhost"]+1)], y1[-1], 
        column_grid["xCentres"][-(column_grid["nGhost"]+2)], y1[-2], 
        column_grid["xCentres"][-1])"""
    
    y1_all = np.concatenate((np.array([y1_ghost_start]), y1, np.array([y1_ghost_end])))

    # Similar calculations for y2 and y3
    """y2_ghost_start = func.quadratic_extrapolation(
        column_grid["xWalls"][column_grid["nGhost"]], y2_inlet, 
        column_grid["xCentres"][column_grid["nGhost"]], y2[0], 
        column_grid["xCentres"][column_grid["nGhost"]+1], y2[1], 
        column_grid["xCentres"][0])
    
    y2_ghost_end = func.quadratic_extrapolation(
        column_grid["xWalls"][-(column_grid["nGhost"]+1)], y2_outlet, 
        column_grid["xCentres"][-(column_grid["nGhost"]+1)], y2[-1], 
        column_grid["xCentres"][-(column_grid["nGhost"]+2)], y2[-2], 
        column_grid["xCentres"][-1])"""

    y2_ghost_start = y2[0] + (column_grid["xCentres"][0] - column_grid["xCentres"][1]) * \
                    (y2[0] - y2_inlet) / (column_grid["xCentres"][column_grid["nGhost"]] - 
                                       column_grid["xWalls"][column_grid["nGhost"]])
    
    y2_ghost_end = y2[-1] - (column_grid["xCentres"][-1] - column_grid["xCentres"][-column_grid["nGhost"]-1]) * \
                  (y2[-1] - y2_outlet) / (column_grid["xCentres"][-(column_grid["nGhost"]+1)] - 
                                       column_grid["xWalls"][-(column_grid["nGhost"]+1)])
    
    y2_all = np.concatenate((np.array([y2_ghost_start]), y2, np.array([y2_ghost_end])))

    """y3_ghost_start = func.quadratic_extrapolation(
        column_grid["xWalls"][column_grid["nGhost"]], y3_inlet, 
        column_grid["xCentres"][column_grid["nGhost"]], y3[0], 
        column_grid["xCentres"][column_grid["nGhost"]+1], y3[1], 
        column_grid["xCentres"][0])
    
    y3_ghost_end = func.quadratic_extrapolation(
        column_grid["xWalls"][-(column_grid["nGhost"]+1)], y3_outlet, 
        column_grid["xCentres"][-(column_grid["nGhost"]+1)], y3[-1], 
        column_grid["xCentres"][-(column_grid["nGhost"]+2)], y3[-2], 
        column_grid["xCentres"][-1])"""
    
    y3_ghost_start = y3[0] + (column_grid["xCentres"][0] - column_grid["xCentres"][1]) * \
                    (y3[0] - y1_inlet) / (column_grid["xCentres"][column_grid["nGhost"]] - 
                                       column_grid["xWalls"][column_grid["nGhost"]])
    
    y3_ghost_end = y3[-1] - (column_grid["xCentres"][-1] - column_grid["xCentres"][-column_grid["nGhost"]-1]) * \
                  (y3[-1] - y3_outlet) / (column_grid["xCentres"][-(column_grid["nGhost"]+1)] - 
                                       column_grid["xWalls"][-(column_grid["nGhost"]+1)])
    
    y3_all = np.concatenate((np.array([y3_ghost_start]), y3, np.array([y3_ghost_end])))
    
    # Temperature ghost cells
    """T_ghost_start = func.quadratic_extrapolation(
        column_grid["xWalls"][column_grid["nGhost"]], T_inlet, 
        column_grid["xCentres"][column_grid["nGhost"]], T[0], 
        column_grid["xCentres"][column_grid["nGhost"]+1], T[1], 
        column_grid["xCentres"][0])
    
    T_ghost_end = func.quadratic_extrapolation(
        column_grid["xWalls"][-(column_grid["nGhost"]+1)], T_outlet, 
        column_grid["xCentres"][-(column_grid["nGhost"]+1)], T[-1], 
        column_grid["xCentres"][-(column_grid["nGhost"]+2)], T[-2], 
        column_grid["xCentres"][-1])"""
    
    T_ghost_start = T[0] + (column_grid["xCentres"][0] - column_grid["xCentres"][1]) * \
                    (T[0] - y1_inlet) / (column_grid["xCentres"][column_grid["nGhost"]] - 
                                       column_grid["xWalls"][column_grid["nGhost"]])
    
    T_ghost_end = T[-1] - (column_grid["xCentres"][-1] - column_grid["xCentres"][-column_grid["nGhost"]-1]) * \
                  (T[-1] - T_outlet) / (column_grid["xCentres"][-(column_grid["nGhost"]+1)] - 
                                       column_grid["xWalls"][-(column_grid["nGhost"]+1)])
    
    T_all = np.concatenate((np.array([T_ghost_start]), T, np.array([T_ghost_end])))

    # Wall temperature ghost cells
    Tw_ghost_start = Tw[0] + (column_grid["xCentres"][0] - column_grid["xCentres"][1]) * \
                     (Tw[0] - Tw_inlet) / (column_grid["xCentres"][column_grid["nGhost"]] - 
                                          column_grid["xWalls"][column_grid["nGhost"]])
    
    Tw_ghost_end = Tw[-1] - (column_grid["xCentres"][-1] - column_grid["xCentres"][-column_grid["nGhost"]-1]) * \
                   (Tw[-1] - Tw_outlet) / (column_grid["xCentres"][-(column_grid["nGhost"]+1)] - 
                                          column_grid["xWalls"][-(column_grid["nGhost"]+1)])
    
    Tw_all = np.concatenate([np.array([Tw_ghost_start]), Tw, np.array([Tw_ghost_end])])

    return P_all, T_all, Tw_all, y1_all, y2_all, y3_all

def calculate_internal_wall_values(P_all, T_all, Tw_all, y1_all, y2_all, y3_all, 
                                   P_inlet, P_outlet, T_inlet, T_outlet, Tw_inlet, Tw_outlet, y1_inlet, y1_outlet, y2_inlet, y2_outlet, 
                                    y3_inlet, y3_outlet, v_inlet, v_outlet, dPdz_inlet, dPdz_outlet, dTwdz_inlet, dTwdz_outlet, bed_properties, column_grid, inlet_values):

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
    dTdz_walls[-1] = 0
    rho_gas_inlet = func.calculate_gas_density(P_all[1], T_all[1])  # [mol/m³]
    Cp_g = func.calculate_gas_heat_capacity()               # [J/mol·K]
    thermal_diffusivity = func.calculate_gas_thermal_conductivity() / (Cp_g * rho_gas_inlet)
    Pe_h = bed_properties["bed_voidage"] / thermal_diffusivity
    v_inlet = inlet_values["velocity"] 
    dTdz_walls[0] = -(v_inlet * Pe_h) * (inlet_values["feed_temperature"] - T_inlet)  # set inlet temperature gradient

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
    avg_density_walls = rho_gas_walls / 1000 * (bed_properties["MW_1"] * y1_walls + bed_properties["MW_2"]*(y2_walls) + 
                                              bed_properties["MW_3"]*(y3_walls)+ bed_properties["MW_4"]*(1-y1_walls-y2_walls-y3_walls))
    mu = func.calculate_gas_viscosity()
    a = 1.75 * (1- bed_properties["bed_voidage"]) * avg_density_walls / (bed_properties["bed_voidage"]**3 * bed_properties["particle_diameter"])
    b = 150 * mu * (1-bed_properties["bed_voidage"])**2 / (bed_properties["bed_voidage"]**3 * bed_properties["particle_diameter"]**2)
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
    
    This function computes the derivatives of the state variables including:
    - Pressure (P)
    - Temperature (T) 
    - Wall temperature (Tw)
    - Component mole fractions (y1, y2, y3)
    - Adsorbed amounts (n1, n2)
    - Flow rates (F) and energy (E) for mass/energy balance tracking
    
    Parameters:
    -----------
    t : float
        Time variable
    results_vector : array
        Current state vector containing all variables
    column_grid : dict
        Grid properties including number of cells and spacing
    bed_properties : dict
        Physical properties of the bed and materials
    inlet_values : dict
        Inlet boundary conditions
    outlet_values : dict
        Outlet boundary conditions
        
    Returns:
    --------
    array
        Derivatives vector for all state variables
    """
    num_cells = column_grid["num_cells"]
    
    # Split results vector into individual variables
    P, T, Tw, y1, y2, y3, n1, n2, F, E = data_prep(results_vector, num_cells)
    
    # =========================================================================
    # BOUNDARY CONDITIONS AND GHOST CELLS
    # =========================================================================
    
    # Apply inlet boundary conditions
    (P_inlet, T_inlet, Tw_inlet, y1_inlet, y2_inlet, y3_inlet, 
     v_inlet, dPdz_inlet, dTwdz_inlet) = inlet_boundary_conditions(
        P, T, Tw, y1, y2, y3, column_grid, bed_properties, inlet_values)

    # Apply outlet boundary conditions
    (P_outlet, T_outlet, Tw_outlet, y1_outlet, y2_outlet, y3_outlet, 
     v_outlet, dPdz_outlet, dTwdz_outlet) = outlet_boundary_conditions(
        P, T, Tw, y1, y2, y3, column_grid, bed_properties, outlet_values)
    
    # Calculate ghost cell values
    P_all, T_all, Tw_all, y1_all, y2_all, y3_all = ghost_cell_calculations(
        P, T, Tw, y1, y2, y3, P_inlet, P_outlet, T_inlet, T_outlet, 
        Tw_inlet, Tw_outlet, y1_inlet, y1_outlet, y2_inlet, y2_outlet, 
        y3_inlet, y3_outlet, column_grid)

    # Calculate internal wall values
    (P_walls, T_walls, Tw_walls, y1_walls, y2_walls, y3_walls, 
     v_walls, dTdz_walls, dTwdz_walls) = calculate_internal_wall_values(
        P_all, T_all, Tw_all, y1_all, y2_all, y3_all,
        P_inlet, P_outlet, T_inlet, T_outlet, Tw_inlet, Tw_outlet,
        y1_inlet, y1_outlet, y2_inlet, y2_outlet, y3_inlet, y3_outlet, 
        v_inlet, v_outlet, dPdz_inlet, dPdz_outlet, dTwdz_inlet, dTwdz_outlet,
        bed_properties, column_grid, inlet_values)
    
    walls = np.concatenate((P_walls, T_walls, Tw_walls, y1_walls, y2_walls, y3_walls, v_walls))

    # =========================================================================
    # MASS TRANSFER AND ADSORPTION KINETICS
    # =========================================================================
    
    # Mass transfer constants
    isotherm_type_1 = bed_properties["isotherm_type_1"]
    isotherm_type_2 = bed_properties["isotherm_type_2"]
    k1 = 0.0002 # * bed_properties["bed_density"] / (1 - bed_properties["bed_voidage"])
    k2 = 0.002 # * bed_properties["bed_density"] / (1 - bed_properties["bed_voidage"])

    # Calculating kinetic constants for component 1
    k1_validation = ( P/(bed_properties["R"]*T)*y1 / func.adsorption_isotherm_1(P, T, y1, y2, y3, n1, bed_properties, isotherm_type_1=isotherm_type_1)[0]
                     * 15 * bed_properties["particle_voidage"] * bed_properties["molecular_diffusivity"] / ( bed_properties["tortuosity"] * (bed_properties["particle_diameter"]/2)**2)
    )

    k2_validation = ( P/(bed_properties["R"]*T)*y2 / func.adsorption_isotherm_2(P, T, y1, y2, bed_properties, isotherm_type_2=isotherm_type_2)[0] 
                        * 15 * bed_properties["particle_voidage"] * bed_properties["molecular_diffusivity"] / ( bed_properties["tortuosity"] * (bed_properties["particle_diameter"]/2)**2)
        )
    # Solid phase balance for adsorbed components
    # ∂q₁/∂t = k₁(q₁* - q₁)
    dn1dt = k1_validation * (func.adsorption_isotherm_1(P, T, y1, y2, y3, n1, bed_properties, isotherm_type_1=isotherm_type_1)[0] - n1) # mol / m3
    deltaH_1 = func.adsorption_isotherm_1(P, T, y1, y2,y3, n1, bed_properties, isotherm_type_1=isotherm_type_1)[1]  # Heat of adsorption (J/mol)


    # ∂q₂/∂t = k₂(q₂* - q₂)
    dn2dt = k2_validation * (func.adsorption_isotherm_2(P, T,y1, y2, bed_properties, isotherm_type_2=isotherm_type_2)[0] - n2)
    deltaH_2 = func.adsorption_isotherm_2(P, T, y1, y2, bed_properties, isotherm_type_2=isotherm_type_2)[1]  # Heat of adsorption (J/mol)


    # =========================================================================
    # TRANSPORT PROPERTIES
    # =========================================================================
    
    # Heat capacities
    Cp_g = func.calculate_gas_heat_capacity() # J/(mol*K)
    Cp_solid = bed_properties["sorbent_heat_capacity"] # J/(kg*K)
    Cp_1 = Cp_g # bed_properties["heat_capacity_1"] # J/(mol*K)
    Cp_2 = Cp_g # bed_properties["heat_capacity_2"] # J/(mol*K)
    Cp_3 = Cp_g # bed_properties["heat_capacity_3"] # J/(mol*K)
    Cp_4 = Cp_g # bed_properties["heat_capacity_4"] # J/(mol*K)

    # Thermal conductivities and heat transfer coefficients
    K_z = func.calculate_gas_thermal_conductivity()
    K_wall = func.calculate_wall_thermal_conductivity()
    h_bed, h_wall = bed_properties["h_bed"], bed_properties["h_wall"]  # Heat transfer coefficients (W/m²*K)
    
    # Axial dispersion coefficient
    D_l = func.calculate_axial_dispersion_coefficient(bed_properties, inlet_values)

    # =========================================================================
    # ENERGY BALANCE - COLUMN TEMPERATURE
    # =========================================================================
    """
    
    a_2 = (
        bed_properties["total_voidage"] * P / (bed_properties["R"] * T) * Cp_g 
        + bed_properties["bed_density"] * Cp_solid
        + (1 - bed_properties["bed_voidage"]) * (Cp_1 * n1 + Cp_2 * n2)
        -  bed_properties["total_voidage"] * P / T
    )
    
    # Column energy balance terms
    conduction_term = (+ K_z * bed_properties["bed_voidage"] *
                      (1 / column_grid["deltaZ"][1:-1]) *
                      (dTdz_walls[1:num_cells+1] - dTdz_walls[:num_cells]))

    convection_term = (- Cp_g * bed_properties["bed_voidage"] / bed_properties["R"] / column_grid["deltaZ"][1:-1] * 
                      (v_walls[1:num_cells+1] * P_walls[1:num_cells+1] - 
                       v_walls[:num_cells] * P_walls[:num_cells]))

    accumulation_term =  (- bed_properties["R"] * T * (1 - bed_properties["bed_voidage"]) * (dn1dt + dn2dt)
                         
                         - T * bed_properties["bed_voidage"] / (column_grid["deltaZ"][1:-1] ) * 
                         (P_walls[1:num_cells+1] * v_walls[1:num_cells+1] / T_walls[1:num_cells+1] - 
                          P_walls[:num_cells] * v_walls[:num_cells] / T_walls[:num_cells]))
    
    adsorption_heat_term = (+ (1 - bed_properties["bed_voidage"]) * 
                           ((np.abs(deltaH_1)) * dn1dt + (np.abs(deltaH_2)) * dn2dt))

    adsorbent_heat_term = (- (1 - bed_properties["bed_voidage"]) * 
                          T * (Cp_1 * dn1dt + Cp_2 * dn2dt))
    
    heat_transfer_term = (-2 * h_bed * (T - Tw) / 
                         (bed_properties["inner_bed_radius"]))
    
    # Combined temperature derivative
    dTdt = (1 / a_2 * (conduction_term + convection_term + accumulation_term + 
                      adsorption_heat_term + adsorbent_heat_term + heat_transfer_term))

    """
    
    # \frac{\partial T}{\partial t} = \frac{1}{a_2}\left[
    # \frac{K_z}{\varepsilon \Delta z}\left(\frac{\partial T}{\partial z}|_{i+1/2} - \frac{\partial T}{\partial z}|_{i-1/2}\right) - \frac{C_{p,g}}{R \Delta z} \left(P_{i+1/2} \ v_{i+1/2} - P_{i-1/2}v_{i-1/2} \right) \right.\\
    # - \frac{C_{p,g}}{R} \left( - \dfrac{(1-\varepsilon)}{\varepsilon}{RT} \left(\frac{\partial q_1}{\partial t}+\frac{\partial q_2}{\partial t}\right) - \frac{1}{\varepsilon} \frac{T}{\Delta z} \left( \frac{P_{i+1/2} \ v_{i+1/2}}{T_{i+1/2}}-\frac{P_{i-1/2} \ v_{i-1/2}}{T_{i-1/2}} \right) \right) \\ \left. + \frac{1-\varepsilon}{\varepsilon} \left(  \Delta H_1 \frac{\partial q_1}{\partial t} +\Delta H_2 \frac{\partial q_2}{\partial t}\right)
    # + \frac{1-\varepsilon}{\varepsilon}  C_{p,ads} T \left( \frac{\partial q_1}{\partial t} + \frac{\partial q_2}{\partial t}\right) + \frac{2h_{bed}}{R_{in}}(T-T_w)\right]"""
    
    # =========================================================================
    # TOTAL ENERGY BALANCE - UGLY VERSION
    # =========================================================================

    a_2 = (
        bed_properties["total_voidage"] * P / (bed_properties["R"] * T) * Cp_g 
        + bed_properties["bed_density"] * Cp_solid
        + (1 - bed_properties["bed_voidage"]) * (Cp_1 * n1 + Cp_2 * n2)
        -  bed_properties["bed_voidage"] * P / T
    )

    # Column energy balance terms
    conduction_term = + ( K_z * bed_properties["bed_voidage"] *
                      (1 / column_grid["deltaZ"][1:-1]) *
                      (dTdz_walls[1:num_cells+1] - dTdz_walls[:num_cells]))

    convection_term = - (bed_properties["bed_voidage"] * 1 / column_grid["deltaZ"][1:-1] 
                         * (P_walls[1:num_cells+1] * v_walls[1:num_cells+1] / bed_properties["R"] - 
                            P_walls[:num_cells] * v_walls[:num_cells] / bed_properties["R"]))
    
    kinetic_term = + ( bed_properties["bed_voidage"] * 1/ column_grid["deltaZ"][1:-1] *
                    (P_walls[1:num_cells+1] * v_walls[1:num_cells+1] - 
                     P_walls[:num_cells] * v_walls[:num_cells]))
    
    conduction_term = + ( bed_properties["bed_voidage"] * K_z / column_grid["deltaZ"][1:-1] *
                        (dTdz_walls[1:num_cells+1] - dTdz_walls[:num_cells]))
    
    dy1dz = (y1_all[1:num_cells+2]- y1_all[0:num_cells+1]) / (column_grid["xCentres"][1:num_cells+2] - column_grid["xCentres"][0:num_cells+1])
    dy2dz = (y2_all[1:num_cells+2]- y2_all[0:num_cells+1]) / (column_grid["xCentres"][1:num_cells+2] - column_grid["xCentres"][0:num_cells+1])
    dy3dz = (y3_all[1:num_cells+2]- y3_all[0:num_cells+1]) / (column_grid["xCentres"][1:num_cells+2] - column_grid["xCentres"][0:num_cells+1])
    dy4dz = - dy1dz - dy2dz - dy3dz

    dispersion_term = + ( bed_properties["bed_voidage"] * D_l / bed_properties["R"] *
                          ( 1 / column_grid["deltaZ"][1:-1] * 
                           (P_walls[1:num_cells+1] * (Cp_1 *dy1dz[1:num_cells+1] + Cp_2 * dy2dz[1:num_cells+1] + 
                                                      Cp_3 * dy3dz[1:num_cells+1] + Cp_4 * dy4dz[1:num_cells+1])
                                                      - P_walls[:num_cells] * (Cp_1 * dy1dz[:num_cells] + Cp_2 * dy2dz[:num_cells] + 
                                                                                 Cp_3 * dy3dz[:num_cells] + Cp_4 * dy4dz[:num_cells]))
    ))

    adsorbent_heat_generation = + (1-bed_properties["bed_voidage"]) * (np.abs(deltaH_1) * dn1dt + np.abs(deltaH_2) * dn2dt)

    accumulation_term = - (bed_properties["bed_voidage"] * (1-bed_properties["bed_voidage"])/bed_properties["total_voidage"]
                           * bed_properties["R"] * T * (dn1dt + dn2dt)

                           + bed_properties["bed_voidage"]**2/bed_properties["total_voidage"] 
                           * T / column_grid["deltaZ"][1:-1] *
                           (P_walls[1:num_cells+1] * v_walls[1:num_cells+1] / T_walls[1:num_cells+1] -
                            P_walls[:num_cells] * v_walls[:num_cells] / T_walls[:num_cells]))
    
    heat_transfer_term = (-2 * h_bed * (T - Tw) / 
                         (bed_properties["inner_bed_radius"]))
    
    dTdt = (1 / a_2 * (conduction_term + convection_term + kinetic_term + dispersion_term +
                      adsorbent_heat_generation + accumulation_term + heat_transfer_term))



    # =========================================================================
    # WALL ENERGY BALANCE
    # =========================================================================
    
    # Wall temperature second derivative
    d2Twdt2 = (1 / column_grid["deltaZ"][1:-1] * 
              (dTwdz_walls[1:num_cells+1] - dTwdz_walls[:num_cells]))
    
    wall_conduction = K_wall * d2Twdt2
    bed_heat_exchange = (2 * bed_properties["inner_bed_radius"] * h_bed * (T - Tw) / 
                        (bed_properties["outer_bed_radius"]**2 - bed_properties["inner_bed_radius"]**2))
    ambient_heat_loss = (-2 * bed_properties["outer_bed_radius"] * h_wall * 
                        (Tw - bed_properties["ambient_temperature"]) / 
                        (bed_properties["outer_bed_radius"]**2 - bed_properties["inner_bed_radius"]**2))
    
    dTwdt = (1 / (bed_properties["wall_heat_capacity"] * bed_properties["wall_density"]) * 
            (wall_conduction + bed_heat_exchange + ambient_heat_loss))
    
    #"""\frac{\partial T_w}{\partial t} =
    #\frac{1}{\rho_w \ C_{p,wall}} \left(\frac{K_{wall}}{\Delta z}\left(\frac{\partial T_w}{\partial z}|_{i+1/2} - \frac{\partial T_w}{\partial z}|_{i-1/2}\right) 
    #+ \frac{2 r_{in} \ h_{bed}}{r_{out}^2-r_{in}^2} \left(T-T_w \right) - \frac{2 r_{out} \ h_{wall}}{r_{out}^2-r_{in}^2} \left(T_w-T_a \right)\right)"""

     # =========================================================================
    # TOTAL MASS BALANCE - PRESSURE
    # =========================================================================
    
    dy1dz = (y1_all[1:num_cells+2]- y1_all[0:num_cells+1]) / (column_grid["xCentres"][1:num_cells+2] - column_grid["xCentres"][0:num_cells+1])
    dy2dz = (y2_all[1:num_cells+2]- y2_all[0:num_cells+1]) / (column_grid["xCentres"][1:num_cells+2] - column_grid["xCentres"][0:num_cells+1])
    dy3dz = (y3_all[1:num_cells+2]- y3_all[0:num_cells+1]) / (column_grid["xCentres"][1:num_cells+2] - column_grid["xCentres"][0:num_cells+1])
    dy4dz = - dy1dz - dy2dz - dy3dz

    thermal_expansion_term = P / T * dTdt
    adsorption_term = (- T * bed_properties["R"] * (1 - bed_properties["bed_voidage"] / bed_properties["total_voidage"]) *
                     (dn1dt + dn2dt))
    convective_term = (-T * bed_properties["bed_voidage"] / bed_properties["total_voidage"] *
                      1 / column_grid["deltaZ"][1:-1] * 
                      (P_walls[1:num_cells+1] * v_walls[1:num_cells+1] / T_walls[1:num_cells+1] - 
                       P_walls[:num_cells] * v_walls[:num_cells] / T_walls[:num_cells]))
    
    dispersion_term = (bed_properties["bed_voidage"] / bed_properties["total_voidage"] * D_l * T / column_grid["deltaZ"][1:-1] *
                          (P_walls[1:num_cells+1] / T_walls[1:num_cells+1] * (dy1dz[1:num_cells+1] + dy2dz[1:num_cells+1] + dy3dz[1:num_cells+1] + dy4dz[1:num_cells+1]) -
                           P_walls[:num_cells] / T_walls[:num_cells] * (dy1dz[:num_cells] + dy2dz[:num_cells] + dy3dz[:num_cells] + dy4dz[:num_cells]))
    )

    dPdt = thermal_expansion_term + adsorption_term + convective_term + dispersion_term
    
    # =========================================================================
    # COMPONENT MASS BALANCES
    # =========================================================================
    
    def calculate_component_balance(yi, yi_walls, yi_all, dnidt=None):
        """Helper function to calculate component mass balance"""
        pressure_effect = -yi / P * dPdt
        temperature_effect = yi / T * dTdt
        
        adsorption_effect = 0
        if dnidt is not None:
            adsorption_effect = (-(1 - bed_properties["bed_voidage"]) / bed_properties["total_voidage"] * 
                               bed_properties["R"] * T / P * dnidt)
        
        convection_effect = (- bed_properties["bed_voidage"] / bed_properties["total_voidage"]  * T / P * 
                           1 / column_grid["deltaZ"][1:-1] * 
                           (P_walls[1:num_cells+1] * v_walls[1:num_cells+1] * yi_walls[1:num_cells+1] / T_walls[1:num_cells+1] - 
                            P_walls[:num_cells] * v_walls[:num_cells] * yi_walls[:num_cells] / T_walls[:num_cells]))
        
        dyidz = (yi_all[1:num_cells+2]- yi_all[0:num_cells+1]) / (column_grid["xCentres"][1:num_cells+2] - column_grid["xCentres"][0:num_cells+1])
        
        dispersion_effect = ( bed_properties["bed_voidage"] / bed_properties["total_voidage"] * D_l * T / P * 1 / column_grid["deltaZ"][1:-1] * 
                           (P_walls[1:num_cells+1] / T_walls[1:num_cells+1] * dyidz[1:num_cells+1] - 
                            P_walls[0:num_cells] / T_walls[0:num_cells] * dyidz[0:num_cells]))
        
        return pressure_effect + temperature_effect + adsorption_effect + convection_effect + dispersion_effect
    
    # Component 1 (adsorbing)
    dy1dt = calculate_component_balance(y1, y1_walls, y1_all, dn1dt)
     #""" \dfrac{\partial y_1}{\partial t} = \dfrac{y_1}{P}\dfrac{\partial P}{\partial t} + \dfrac{y_1}{T}\dfrac{\partial T}{\partial t} 
    #- \dfrac{(1-\varepsilon)}{\varepsilon}\frac{RT}{P}\frac{\partial q_1}{\partial t} - \frac{1}{\varepsilon}\frac{T}{P}\frac{1}{\Delta z} 
    #\left( \frac{P_{i+1/2} \ y_{1,i+1/2} \ v_{i+1/2}}{T_{i+1/2}}-\frac{P_{i-1/2} \ y_{1,i-1/2} \ v_{i-1/2}}{T_{i-1/2}} \right) \\ 
    #+ D_l \  \frac{T}{P} \ \frac{1}{\Delta z} \left( \frac{P_{i+1/2}}{T_{i+1/2}} \ \frac{y_{1,i+1} - y_{1,i}}{\Delta z} - \frac{P_{i-1/2}}{T_{i-1/2}} \ \frac{y_{1,i} - y_{1,i-1}}{\Delta z}\right)"""
    
    
    # Component 2 (adsorbing)
    dy2dt = calculate_component_balance(y2, y2_walls, y2_all, dn2dt)
     #"""\dfrac{\partial y_2}{\partial t} = \dfrac{y_2}{P}\dfrac{\partial P}{\partial t} + \dfrac{y_2}{T}\dfrac{\partial T}{\partial t} - \dfrac{(1-\varepsilon)}{\varepsilon}\frac{RT}{P}\frac{\partial q_2}{\partial t} 
    #- \frac{1}{\varepsilon}\frac{T}{P}\frac{1}{\Delta z} \left( \frac{P_{i+1/2} \ y_{2,i+1/2} \ v_{i+1/2}}{T_{i+1/2}}-\frac{P_{i-1/2} \ y_{2,i-1/2} \ v_{i-1/2}}{T_{i-1/2}} \right)
    #\\  + D_l \  \frac{T}{P} \ \frac{1}{\Delta z} \left( \frac{P_{i+1/2}}{T_{i+1/2}} \ \frac{y_{2,i+1} - y_{2,i}}{\Delta z} - \frac{P_{i-1/2}}{T_{i-1/2}} \ \frac{y_{2,i} - y_{2,i-1}}{\Delta z}\right)"""
    
    
    # Component 3 (non-adsorbing)
    dy3dt = calculate_component_balance(y3, y3_walls, y3_all)
        #""" \dfrac{\partial y_3}{\partial t} = \dfrac{y_3}{P}\dfrac{\partial P}{\partial t} + \dfrac{y_3}{T}\dfrac{\partial T}{\partial t} 
    #- \frac{1}{\varepsilon}\frac{T}{P}\frac{1}{\Delta z} \left( \frac{P_{i+1/2} \ y_{3,i+1/2} \ v_{i+1/2}}{T_{i+1/2}}-\frac{P_{i-1/2} \ y_{3,i-1/2} \ v_{i-1/2}}{T_{i-1/2}} \right)
    #\\ + D_l \  \frac{T}{P} \ \frac{1}{\Delta z} \left( \frac{P_{i+1/2}}{T_{i+1/2}} \ \frac{y_{3,i+1} - y_{3,i}}{\Delta z} - \frac{P_{i-1/2}}{T_{i-1/2}} \ \frac{y_{3,i} - y_{3,i-1}}{\Delta z}\right)"""

    # Component 4 (non-adsorbing)
    dy4dt = - (dy1dt + dy2dt + dy3dt)



    # \dfrac{\partial P}{\partial t} = \dfrac{P}{T}\dfrac{\partial T}{\partial t} - \dfrac{(1-\varepsilon)}{\varepsilon}{RT} \left(\frac{\partial q_1}{\partial t}+\frac{\partial q_2}{\partial t}\right)
    #- \frac{1}{\varepsilon} \frac{T}{\Delta z} \left( \frac{P_{i+1/2} \ v_{i+1/2}}{T_{i+1/2}}-\frac{P_{i-1/2} \ v_{i-1/2}}{T_{i-1/2}} \right)"""
   
    # =========================================================================
    # MASS AND ENERGY BALANCE TRACKING
    # =========================================================================
    
    # Inlet and outlet calculations for mass balance error
    dF1dt = bed_properties["bed_voidage"] * bed_properties["inner_bed_radius"]**2 * np.pi / (bed_properties["R"] * T_walls[0]) * v_walls[0]*P_walls[0] * y1_walls[0]
    dF2dt = bed_properties["bed_voidage"] * bed_properties["inner_bed_radius"]**2 * np.pi / (bed_properties["R"] * T_walls[0]) * v_walls[0]*P_walls[0] * y2_walls[0]
    dF3dt = bed_properties["bed_voidage"] * bed_properties["inner_bed_radius"]**2 * np.pi / (bed_properties["R"] * T_walls[0]) * v_walls[0]*P_walls[0] * y3_walls[0]
    dF4dt = bed_properties["bed_voidage"] * bed_properties["inner_bed_radius"]**2 * np.pi / (bed_properties["R"] * T_walls[0]) * v_walls[0]*P_walls[0] * (1 - y1_walls[0]-y2_walls[0]-y3_walls[0])


    dF5dt = bed_properties["bed_voidage"] * bed_properties["inner_bed_radius"]**2 * np.pi / (bed_properties["R"] * T_walls[-1]) * v_walls[-1]*P_walls[-1] * y1_walls[-1]
    dF6dt = bed_properties["bed_voidage"] * bed_properties["inner_bed_radius"]**2 * np.pi / (bed_properties["R"] * T_walls[-1]) * v_walls[-1]*P_walls[-1] * y2_walls[-1]
    dF7dt = bed_properties["bed_voidage"] * bed_properties["inner_bed_radius"]**2 * np.pi / (bed_properties["R"] * T_walls[-1]) * v_walls[-1]*P_walls[-1] * y3_walls[-1]
    dF8dt = bed_properties["bed_voidage"] * bed_properties["inner_bed_radius"]**2 * np.pi / (bed_properties["R"] * T_walls[-1]) * v_walls[-1]*P_walls[-1] * (1 - y1_walls[-1]-y2_walls[-1]-y3_walls[-1])

    dFdt = np.array([dF1dt, dF2dt, dF3dt, dF4dt, dF5dt, dF6dt, dF7dt, dF8dt])

    #Inlet and outlet calculations for energy balance error
    dE1dt = (bed_properties["bed_voidage"] * bed_properties["inner_bed_radius"]**2 * np.pi * Cp_g * v_walls[0] * T_walls[0] * P_walls[0] / (bed_properties["R"] * T_walls[0]))
    dE2dt = (bed_properties["bed_voidage"] * bed_properties["inner_bed_radius"]**2 * np.pi * Cp_g * v_walls[-1] * T_walls[-1] * P_walls[-1] / (bed_properties["R"] * T_walls[-1]))
    dE3dt = np.sum(2 * np.pi * bed_properties["outer_bed_radius"] * h_wall * (Tw - bed_properties["ambient_temperature"]) * column_grid["deltaZ"][1:-1])
    dEdt = np.array([dE1dt, dE2dt, dE3dt])

    # Combine derivatives into a single vector
    derivatives = np.concatenate([dPdt, dTdt, dTwdt, dy1dt, dy2dt, dy3dt, dn1dt, dn2dt, dFdt, dEdt]) 
    
    return derivatives

def final_wall_values(column_grid, bed_properties, inlet_values, outlet_values, output_matrix):
    """
    Calculate wall values for each time step in the adsorption column simulation.
    
    Returns:
        Arrays with shape (n_walls, n_timesteps) for each wall variable.
    """
    import numpy as np

    # Initialize lists to accumulate time series
    P_walls_ = []
    T_walls_ = []
    Tw_walls_ = []
    y1_walls_ = []
    y2_walls_ = []
    y3_walls_ = []
    v_walls_ = []

    num_cells = column_grid["num_cells"]
    num_timesteps = output_matrix.t.shape[0]

    for t in range(num_timesteps):
        # Get state vector at timestep t

        # Unpack variables from the state
        P, T, Tw, y1, y2, y3, n1, n2, F, E = data_prep(output_matrix.y[:, t], num_cells)

        # Boundary conditions
        (P_inlet, T_inlet, Tw_inlet, y1_inlet, y2_inlet, y3_inlet, 
         v_inlet, dPdz_inlet, dTwdz_inlet) = inlet_boundary_conditions(
             P, T, Tw, y1, y2, y3, column_grid, bed_properties, inlet_values)

        (P_outlet, T_outlet, Tw_outlet, y1_outlet, y2_outlet, y3_outlet, 
         v_outlet, dPdz_outlet, dTwdz_outlet) = outlet_boundary_conditions(
             P, T, Tw, y1, y2, y3, column_grid, bed_properties, outlet_values)

        # Ghost cells
        P_all, T_all, Tw_all, y1_all, y2_all, y3_all = ghost_cell_calculations(
            P, T, Tw, y1, y2, y3, P_inlet, P_outlet, T_inlet, T_outlet, 
            Tw_inlet, Tw_outlet, y1_inlet, y1_outlet, y2_inlet, y2_outlet, 
            y3_inlet, y3_outlet, column_grid)

        # Wall values for this timestep
        (P_walls, T_walls, Tw_walls, y1_walls, y2_walls, y3_walls, 
         v_walls, dTdz_walls, dTwdz_walls) = calculate_internal_wall_values(
            P_all, T_all, Tw_all, y1_all, y2_all, y3_all,
            P_inlet, P_outlet, T_inlet, T_outlet, Tw_inlet, Tw_outlet,
            y1_inlet, y1_outlet, y2_inlet, y2_outlet, y3_inlet, y3_outlet, 
            v_inlet, v_outlet, dPdz_inlet, dPdz_outlet, dTwdz_inlet, dTwdz_outlet,
            bed_properties, column_grid, inlet_values)

        # Append current timestep wall values
        P_walls_.append(P_walls)
        T_walls_.append(T_walls)
        Tw_walls_.append(Tw_walls)
        y1_walls_.append(y1_walls)
        y2_walls_.append(y2_walls)
        y3_walls_.append(y3_walls)
        v_walls_.append(v_walls)

    # Convert to arrays of shape (n_walls, n_timesteps)
    return (
        np.array(P_walls_).T,
        np.array(T_walls_).T,
        np.array(Tw_walls_).T,
        np.array(y1_walls_).T,
        np.array(y2_walls_).T,
        np.array(y3_walls_).T,
        np.array(v_walls_).T
    )