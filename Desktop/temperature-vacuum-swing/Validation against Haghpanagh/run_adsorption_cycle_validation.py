
import numpy as np
from scipy.integrate import solve_ivp
from additional_functions_validation import create_non_uniform_grid, adsorption_isotherm_1, adsorption_isotherm_2, total_mass_balance_error, CO2_mass_balance_error, energy_balance_error, create_plot, create_combined_plot
from scipy import integrate
import time
import matplotlib.pyplot as plt
import tvsa_adsorption_column_validation as column

"""This script sets up the initial conditions and parameters for an adsorption column model simulation.
It defines the column grid, bed properties, inlet and outlet conditions, and initializes the state variables for the simulation.
The components are (1) CO2, (2) H2O, (3) N2 and (4) O2."""


input_parameters = {

}


# Define fixed bed properties
bed_properties_pini = {
    #Column dimensions and properties
    "bed_length": 0.286,  # Example value for bed length in meters
    "inner_bed_radius": 0.00945,
    "outer_bed_radius": 0.0127,  # Example value for bed diameter in meters
    "bed_voidage": 0.456,  # Example value for bed voidage
    "bed_density": 435,  # Example value for bed density in kg/m^3
    "column_area": 0.00945**2 * np.pi,  # Cross-sectional area of the column
    "sorbent_density": 900, #kg/m3, 55.4 (Haghpanagh et al. 2013)
    
    # Properties and constants
    "particle_voidage": 0.59,  # Example value for particle voidage
    "tortuosity": 3,  # Example value for tortuosity
    "molecular_diffusivity": 1.6e-5,  # Example value for molecular diffusivity in m²/s
    "particle_diameter": 3e-3,  # Example value for particle diameter in meters
    "wall_density": 2700,  # Example value for wall density in kg/m^3 (Haghpanagh et al. 2013)
    "R": 8.314,  # Universal gas constant in J/(mol*K)
    "sorbent_mass": 0.286 * 0.00945**2 * np.pi* 435,  # Example value for sorbent mass in kg
    "sorbent_heat_capacity": 1040, # J /kg/K, # Example value for solid heat capacity
    "wall_heat_capacity": 902, # J /kg/K, # Example value for wall heat capacity (Haghpanagh et al. 2013)
    "heat_capacity_1": 840 * 44.01 / 1000,  # J / kgK / (kg/mol) J/mol K * kg/m3 Example value for adsorbed phase heat capacity of component 1 (CO2) 
    "heat_capacity_2":30,  # Example value for adsorbed phase heat capacity of component 2 (H2O)
    "mass_transfer_1": 0.0002,  # Example value for mass transfer coefficient of component 1 (CO2) in s-1
    "mass_transfer_2": 0.002,  # Example value for mass transfer coefficient of component 2 (H2O) in s-1
    "MW_1": 44.01,  # Molecular weight of component 1 (CO2) in g/mol
    "MW_2": 18.02,  # Molecular weight of component 2 (H2O) in g/mol
    "MW_3": 28.02,  # Molecular weight of component 3 (N2) in g/mol
    "MW_4": 32.00,  # Molecular weight of component 4 (O2) in g/mol
    "mu": 1.78e-5,  # Example value for feed viscosity in Pa.s
    "isotherm_type_1": "Dual_site Langmuir",
    "isotherm_type_2": "None",  # Example value for isotherm type for component 2 (H2O)
}

bed_properties_stampi_bombelli_analysis = {
    "bed_voidage": 0.4,  # Example value for bed voidage
    "particle_diameter": 0.0075,  # Example value for particle diameter in meters
    "inner_bed_diameter": 0.08,
    "outer_bed_diameter": 0.082,  # Example value for bed diameter in meters
    "column_area": 0.08**2 * np.pi / 4,  # Cross-sectional area of the column
    "R": 8.314,  # Universal gas constant in J/(mol*K)
    "T_column": 298,  # Column temperature in Kelvin
    "bed_density": 55.4,  # Example value for bed density in kg/m^3
    "wall_density": 7800,  # Example value for wall density in kg/m^3 (Haghpanagh et al. 2013)
    "solid_density": 1130,  # Example value for solid density in kg/m^3
    "sorbent_mass": 0.4* 0.08**2 * np.pi / 4 * 0.01 * 55.4,  # Example value for sorbent mass in kg
    "bed_length": 0.01,  # Example value for bed length in meters
    "wall_heat_capacity": 502, # J /kg/K, # Example value for wall heat capacity (Haghpanagh et al. 2013)
    "wall_conductivity" : 16, # K_w, J/m.K.s
    "sorbent_density": 55.4, #kg/m3, 55.4 (Haghpanagh et al. 2013)
    "sorbent_heat_capacity": 2070, # J/kg/K, # Example value for solid heat capacity
    "heat_transfer_coefficient": 3,  # Example value for heat transfer coefficient in W/(m^2*K)
    "heat_transfer_coefficient_wall": 26,  # Example value for wall heat transfer coefficient in W/(m^2*K)
    # "adsorption_heat_1": -57,  # Example value for adsorption heat of component 1 (CO2) in kJ/mol
    "heat_capacity_1": 42.46 ,  # Example value for adsorbed phase heat capacity of component 1 (CO2) in J/(mol*K) (Haghpanagh et al. 2013)
    "mass_transfer_1": 0.0002,  # Example value for mass transfer coefficient of component 1 (CO2) in s-1
    # "adsorption_heat_2": -49,  # Example value for adsorption heat of component 2 (H2O) in kJ/mol
    "heat_capacity_2": 73.1,  # Example value for adsorbed phase heat capacity of component 2 (H2O) in J/(mol*K) (Haghpanagh et al. 2013)
    "mass_transfer_2": 0.002,  # Example value for mass transfer coefficient of component 2 (H2O) in s-1
    "MW_1": 44.01,  # Molecular weight of component 1 (CO2) in g/mol
    "MW_2": 18.02,  # Molecular weight of component 2 (H2O) in g/mol
    "MW_3": 28.02,  # Molecular weight of component 3 (N2) in g/mol
    "MW_4": 32.00,  # Molecular weight of component 4 (O2) in g/mol

}


# Define bed inlet values (subject to variation)
inlet_values = {
    "inlet_type": "mass_flow",
    "velocity": 100 / 60 / 1e6 / bed_properties_pini["column_area"] / bed_properties_pini["bed_voidage"],  # Example value for superficial velocity in m/s
    "rho_gas": 1.13,  # Example value for feed density in kg/m^3
    "feed_volume_flow": 1.6667e-6,  # cm³/min to m³/s
    "feed_mass_flow": (0.01 * float(bed_properties_pini["column_area"]) * 1.13),  # Example value for feed mass flow in kg/s
    "feed_temperature": 293.15,  # Example value for feed temperature in Kelvin
    "feed_pressure": 101325,  # Example value for feed pressure in Pa
    "y1_feed_value": 0.999999,  # Example value for feed mole fraction
    "y2_feed_value": 1e-6,  # Example value for feed mole fraction
    "y3_feed_value": 0,  # Example value for feed mole fraction
}

# Define bed outlet values (subject to variation)
outlet_values = {
    "outlet_type": "pressure",
    "outlet_pressure": 101325,  # Example value for outlet pressure in Pa
    "outlet_temperature": 293.15,  # Example value for outlet temperature in Kelvin
}
column_grid = create_non_uniform_grid(bed_properties_pini)

# Initial values for the column
P = np.ones(column_grid["num_cells"]) * 101325  # Example pressure vector in Pa
T = np.ones(column_grid["num_cells"]) * 292  # Example temperature vector in K
Tw = np.ones(column_grid["num_cells"]) * 292  # Example wall temperature vector in K
y1 = np.ones(column_grid["num_cells"]) * 1e-6  # Example mole fraction of component 1
y2 = np.ones(column_grid["num_cells"]) * 1e-6 # Example mole fraction of component 2
y3 = np.ones(column_grid["num_cells"]) * 0.95  # Example mole fraction of component 3
n1 = adsorption_isotherm_1(P, T, y1, y2, y3, 1e-6, bed_properties=bed_properties_pini, isotherm_type=bed_properties_pini["isotherm_type_1"])[0]  # Example concentration vector for component 1 in mol/m^3
n2 = adsorption_isotherm_2(P, T, y2, bed_properties=bed_properties_pini, isotherm_type=bed_properties_pini["isotherm_type_2"])[0]# Example concentration vector for component 2 in mol/m^3
F = np.zeros(8)
E = np.zeros(3)  # Additional variables (e.g., flow rates, mass balances)
initial_conditions = np.concatenate([P, T, Tw, y1, y2, y3, n1, n2, F, E])

# Running simulation! ======================================================================================================

# Implement solver
t_span = [0, 3000]  # Time span for the ODE solver
rtol = 1e-4
atol_P = 1e-2 * np.ones(len(P))
atol_T = 1e-3 * np.ones(len(T))
atol_Tw = 1e-4 * np.ones(len(Tw))
atol_y1 = 1e-7 * np.ones(len(y1))
atol_y2 = 1e-7 * np.ones(len(y2))
atol_y3 = 1e-7 * np.ones(len(y3))
atol_n1 = 1e-2 * np.ones(len(n1))
atol_n2 = 1e-2 * np.ones(len(n2))
atol_F = 1e-4 * np.ones(len(F))
atol_E = 1e-4 * np.ones(len(E))
atol_array = np.concatenate([atol_P, atol_T, atol_Tw, atol_y1, atol_y2, atol_y3, atol_n1, atol_n2, atol_F, atol_E])
t0=time.time()
def ODE_func(t, results_vector):
    return column.ODE_calculations(t, results_vector=results_vector, column_grid=column_grid, bed_properties=bed_properties_pini, inlet_values=inlet_values, outlet_values=outlet_values)
output_matrix = solve_ivp(ODE_func, t_span, initial_conditions, method='BDF', rtol=rtol, atol=atol_array)
t1=time.time()
total_time = t1 - t0
    
#print(output_matrix)
#print(output_matrix.t)

P_result = output_matrix.y[0:column_grid["num_cells"]]
T_result = output_matrix.y[column_grid["num_cells"]:2*column_grid["num_cells"]]
Tw_result = output_matrix.y[2*column_grid["num_cells"]:3*column_grid["num_cells"]]
y1_result = output_matrix.y[3*column_grid["num_cells"]:4*column_grid["num_cells"]]
y2_result = output_matrix.y[4*column_grid["num_cells"]:5*column_grid["num_cells"]]
y3_result = output_matrix.y[5*column_grid["num_cells"]:6*column_grid["num_cells"]]
n1_result = output_matrix.y[6*column_grid["num_cells"]:7*column_grid["num_cells"]]
n2_result = output_matrix.y[7*column_grid["num_cells"]:8*column_grid["num_cells"]]
F_result = output_matrix.y[8*column_grid["num_cells"]:8*column_grid["num_cells"]+8]
E_result = output_matrix.y[8*column_grid["num_cells"]+8:]
time = output_matrix.t

print("Mass balance error:", total_mass_balance_error(F_result, P_result, T_result, n1_result, n2_result, time, bed_properties_pini, column_grid))
print("CO2 mass balance error:", CO2_mass_balance_error(F_result, P_result, T_result, y1_result, n1_result, time, bed_properties_pini, column_grid))
print("Energy balance error:", energy_balance_error(E_result, T_result, P_result, y1_result, y2_result, y3_result, n1_result, n2_result, time, bed_properties_pini, column_grid))


# Calculate the wall values

def final_wall_values(column_grid, bed_properties, output_matrix):
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
        P, T, Tw, y1, y2, y3, n1, n2, F, E = column.data_prep(output_matrix.y[:, t], num_cells)

        # Boundary conditions
        (P_inlet, T_inlet, Tw_inlet, y1_inlet, y2_inlet, y3_inlet, 
         v_inlet, dPdz_inlet, dTwdz_inlet) = column.inlet_boundary_conditions(
             P, T, Tw, y1, y2, y3, column_grid, bed_properties, inlet_values)

        (P_outlet, T_outlet, Tw_outlet, y1_outlet, y2_outlet, y3_outlet, 
         v_outlet, dPdz_outlet, dTwdz_outlet) = column.outlet_boundary_conditions(
             P, T, Tw, y1, y2, y3, column_grid, bed_properties, outlet_values)

        # Ghost cells
        P_all, T_all, Tw_all, y1_all, y2_all, y3_all = column.ghost_cell_calculations(
            P, T, Tw, y1, y2, y3, P_inlet, P_outlet, T_inlet, T_outlet, 
            Tw_inlet, Tw_outlet, y1_inlet, y1_outlet, y2_inlet, y2_outlet, 
            y3_inlet, y3_outlet, column_grid)

        # Wall values for this timestep
        (P_walls, T_walls, Tw_walls, y1_walls, y2_walls, y3_walls, 
         v_walls, dTdz_walls, dTwdz_walls) = column.calculate_internal_wall_values(
            P_all, T_all, Tw_all, y1_all, y2_all, y3_all,
            P_inlet, P_outlet, T_inlet, T_outlet, Tw_inlet, Tw_outlet,
            y1_inlet, y1_outlet, y2_inlet, y2_outlet, y3_inlet, y3_outlet, 
            v_inlet, v_outlet, dPdz_inlet, dPdz_outlet, dTwdz_inlet, dTwdz_outlet,
            bed_properties, column_grid)

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

(P_walls_result, T_walls_result, Tw_walls_result,
 y1_walls_result, y2_walls_result, y3_walls_result,
 v_walls_result) = final_wall_values(column_grid, bed_properties_pini, output_matrix)

# Create the combined plot
create_combined_plot(time, T_result, P_result, y1_result, n1_result, y1_walls_result, v_walls_result, bed_properties_pini)
