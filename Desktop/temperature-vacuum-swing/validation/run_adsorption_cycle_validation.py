
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


# Define fixed bed properties
def create_fixed_properties():
    bed_properties = {
    #Column dimensions and properties
    "bed_length":0.286,  # Example value for bed length in meters
    "inner_bed_radius": 0.00945,
    "outer_bed_radius": 0.0127,  # Example value for bed diameter in meters
    "bed_voidage": 0.456,  # Example value for bed voidage
    "total_voidage": 0.456+(1-0.456)*0.59,  # Example value for total voidage
    "bed_density": 435,  # Example value for bed density in kg/m^3
    "column_area": 0.00945**2 * np.pi,  # Cross-sectional area of the column
    "sorbent_density": 900, #kg/m3, 55.4 (Haghpanagh et al. 2013)
    "ambient_temperature": 293.15,  # Example value for ambient temperature in Kelvin
    
    # Properties and constants
    "particle_voidage": 0.59,  # Example value for particle voidage
    "tortuosity": 3,  # Example value for tortuosity
    "molecular_diffusivity": 0.605e-5,  # Example value for molecular diffusivity in m²/s
    "particle_diameter": 2e-3,  # Example value for particle diameter in meters
    "wall_density": 2700,  # Example value for wall density in kg/m^3 (Haghpanagh et al. 2013)
    "R": 8.314,  # Universal gas constant in J/(mol*K)
    "sorbent_mass": 0.286 * 0.00945**2 * np.pi* 435,  # Example value for sorbent mass in kg
    "sorbent_heat_capacity": 1040, # J /kg/K, # Example value for solid heat capacity
    "wall_heat_capacity": 902, # J /kg/K, # Example value for wall heat capacity (Haghpanagh et al. 2013)
    "heat_capacity_1": 840 * 44.01 / 1000,  # J / kgK / (kg/mol) J/mol K * kg/m3 Example value for adsorbed phase heat capacity of component 1 (CO2) 
    "heat_capacity_2": 30,  # Example value for adsorbed phase heat capacity of component 2 (H2O)
    "mass_transfer_1": 0.0002,  # Example value for mass transfer coefficient of component 1 (CO2) in s-1
    "mass_transfer_2": 0.002,  # Example value for mass transfer coefficient of component 2 (H2O) in s-1
    "h_bed": 50,  # 140,  # Example value for bed heat transfer coefficient in W/m²·K
    "h_wall": 7,  # 20,  # Example value for wall heat transfer coefficient in W
    "MW_1": 44.01,  # Molecular weight of component 1 (CO2) in g/mol
    "MW_2": 18.02,  # Molecular weight of component 2 (H2O) in g/mol
    "MW_3": 28.02,  # Molecular weight of component 3 (N2) in g/mol
    "MW_4": 32.00,  # Molecular weight of component 4 (O2) in g/mol
    "mu": 1.78e-5,  # Example value for feed viscosity in Pa.s
    "isotherm_type_1": "Dual_site Langmuir",
    "isotherm_type_2": "None",  # Example value for isotherm type for component 2 (H2O)

    #Reference values for scaling
    "P_ref": 101325,  # Reference pressure in Pa
    "T_ref": 298.15,  # Reference temperature in K
    "n_ref": 3000,  # Reference adsorbed amount in mol/m^3

    "t_tot": 100,
}
    column_grid = create_non_uniform_grid(bed_properties)

    # Initial values for the column
    P = np.ones(column_grid["num_cells"]) * 101325  # Example pressure vector in Pa
    T = np.ones(column_grid["num_cells"]) * 292  # Example temperature vector in K
    Tw = np.ones(column_grid["num_cells"]) * 292  # Example wall temperature vector in K
    y1 = np.ones(column_grid["num_cells"]) * 0.02  # Example mole fraction of component 1
    y2 = np.ones(column_grid["num_cells"]) * 1e-6 # Example mole fraction of component 2
    y3 = np.ones(column_grid["num_cells"]) * 0.95  # Example mole fraction of component 3
    n1 = adsorption_isotherm_1(P, T, y1, y2, y3, 1e-6, bed_properties=bed_properties, isotherm_type_1=bed_properties["isotherm_type_1"])[0]  # Example concentration vector for component 1 in mol/m^3
    n2 = adsorption_isotherm_2(P, T, y2, bed_properties=bed_properties, isotherm_type=bed_properties["isotherm_type_2"])[0]# Example concentration vector for component 2 in mol/m^3
    F = np.zeros(8)
    E = np.zeros(3)  # Additional variables (e.g., flow rates, mass balances)
    initial_conditions = np.concatenate([P/bed_properties["P_ref"], T/bed_properties["T_ref"], Tw/ bed_properties["T_ref"], y1, y2, y3, n1/bed_properties["n_ref"], n2/bed_properties["n_ref"], F, E])

    rtol = 1e-5

    atol_P = 1e-5 * np.ones(len(P))
    atol_T = 1e-4 * np.ones(len(T))
    atol_Tw = 1e-5 * np.ones(len(Tw))
    atol_y1 = 1e-8 * np.ones(len(y1))
    atol_y2 = 1e-8 * np.ones(len(y2))
    atol_y3 = 1e-8 * np.ones(len(y3))
    atol_n1 = 1e-3 * np.ones(len(n1))
    atol_n2 = 1e-3 * np.ones(len(n2))
    atol_F = 1e-4 * np.ones(len(F))
    atol_E = 1e-4 * np.ones(len(E))
    atol_array = np.concatenate([atol_P, atol_T, atol_Tw, atol_y1, atol_y2, atol_y3, atol_n1, atol_n2, atol_F, atol_E])
    

    return bed_properties, column_grid, initial_conditions, rtol, atol_array

# Define bed inlet values (subject to variation)
def define_boundary_conditions(bed_properties):
    inlet_values = {
        "inlet_type": "mass_flow",
        "velocity": 100 / 60 / 1e6 / bed_properties["column_area"] / bed_properties["bed_voidage"],  # Example value for interstitial velocity in m/s
        "rho_gas": 1.13,  # Example value for feed density in kg/m^3
        "feed_volume_flow": 1.6667e-6,  # cm³/min to m³/s
        "feed_mass_flow": (0.01 * float(bed_properties["column_area"]) * 1.13),  # Example value for feed mass flow in kg/s
        "feed_temperature": 293.15,  # Example value for feed temperature in Kelvin
        "feed_pressure": 101325,  # Example value for feed pressure in Pa
        "y1_feed_value": 0.9999,  # Example value for feed mole fraction
        "y2_feed_value": 1e-6,  # Example value for feed mole fraction
        "y3_feed_value": 1e-6,  # Example value for feed mole fraction
    }
        # Define bed outlet values (subject to variation)
    outlet_values = {
        "outlet_type": "pressure",
        "outlet_pressure": 101325,  # Example value for outlet pressure in Pa
        "outlet_temperature": 293.15,  # Example value for outlet temperature in Kelvin
        }
    return inlet_values, outlet_values


# Running simulation! ======================================================================================================

P_result = []
T_result = []
Tw_result = [] 
y1_result = []
y2_result = []
y3_result = []
n1_result = []
n2_result = []
F_result = []
E_result = []

# Implement solver
bed_properties, column_grid, initial_conditions, rtol, atol_array = create_fixed_properties()
inlet_values, outlet_values = define_boundary_conditions(bed_properties)
t_span = [0, 3000]  # Time span for the ODE solver

t0=time.time()
def ODE_func(t, results_vector,):
    return column.ODE_calculations(t, results_vector=results_vector, column_grid=column_grid, bed_properties=bed_properties, inlet_values=inlet_values, outlet_values=outlet_values)
output_matrix = solve_ivp(ODE_func, t_span, initial_conditions, method='BDF', rtol=rtol, atol=atol_array)
t1=time.time()
total_time = t1 - t0
    

P_result = output_matrix.y[0:column_grid["num_cells"]] * bed_properties["P_ref"]
T_result = output_matrix.y[column_grid["num_cells"]:2*column_grid["num_cells"]] * bed_properties["T_ref"]
Tw_result = output_matrix.y[2*column_grid["num_cells"]:3*column_grid["num_cells"]] * bed_properties["T_ref"]
y1_result = output_matrix.y[3*column_grid["num_cells"]:4*column_grid["num_cells"]]
y2_result = output_matrix.y[4*column_grid["num_cells"]:5*column_grid["num_cells"]]
y3_result = output_matrix.y[5*column_grid["num_cells"]:6*column_grid["num_cells"]]
n1_result = output_matrix.y[6*column_grid["num_cells"]:7*column_grid["num_cells"]] * bed_properties["n_ref"]
n2_result = output_matrix.y[7*column_grid["num_cells"]:8*column_grid["num_cells"]] * bed_properties["n_ref"]
F_result = output_matrix.y[8*column_grid["num_cells"]:8*column_grid["num_cells"]+8]
E_result = output_matrix.y[8*column_grid["num_cells"]+8:]
time = output_matrix.t

print("Mass balance error:", total_mass_balance_error(F_result, P_result, T_result, n1_result, n2_result, time, bed_properties, column_grid))
print("CO2 mass balance error:", CO2_mass_balance_error(F_result, P_result, T_result, y1_result, n1_result, time, bed_properties, column_grid))
print("Energy balance error:", energy_balance_error(E_result, T_result, P_result, y1_result, y2_result, y3_result, n1_result, n2_result, Tw_result, time, bed_properties, column_grid))
print("Total simulation time:", total_time, "seconds")
print("Duration of simulation:", time[-1], "seconds")

# Calculate the exit column values
(P_walls_result, T_walls_result, Tw_walls_result,
 y1_walls_result, y2_walls_result, y3_walls_result,
 v_walls_result) = column.final_wall_values(column_grid, bed_properties, inlet_values, outlet_values, output_matrix)

# Create the combined plot
#create_combined_plot(time, T_result, P_result, y1_result, n1_result, y1_walls_result, v_walls_result, bed_properties)

#create_plot(time, T_result, "Temperature evolution", "Temperature")
#create_plot(time, n1_result, "Adsorbed phase", "CO2 adsorbed (mol/m^3)")
