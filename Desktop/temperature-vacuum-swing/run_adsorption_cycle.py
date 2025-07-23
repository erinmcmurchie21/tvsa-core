
import numpy as np
from scipy.integrate import solve_ivp
from additional_functions import create_non_uniform_grid, adsorption_isotherm_1, adsorption_isotherm_2, quadratic_extrapolation, quadratic_extrapolation_derivative, quadratic_extrapolation_derivative_nonzero, mass_balance_error, energy_balance_error
from scipy import integrate
import time
import matplotlib.pyplot as plt
from tvsa_adsorption_column import ODE_calculations

"""This script sets up the initial conditions and parameters for an adsorption column model simulation.
It defines the column grid, bed properties, inlet and outlet conditions, and initializes the state variables for the simulation.
The components are (1) CO2, (2) H2O, (3) N2 and (4) O2."""


column_grid = create_non_uniform_grid()

# Define fixed bed properties
bed_properties = {
    "bed_voidage": 0.4,  # Example value for bed voidage
    "particle_diameter": 0.0075,  # Example value for particle diameter in meters
    "inner_bed_diameter": 0.08,
    "outer_bed_diameter": 0.082,  # Example value for bed diameter in meters
    "column_area": 0.08**2 * np.pi / 4,  # Cross-sectional area of the column
    "R": 8.314,  # Universal gas constant in J/(mol*K)
    "T_column": 298,  # Column temperature in Kelvin
    "rho_bed": 55.4,  # Example value for bed density in kg/m^3
    "wall_density": 7800,  # Example value for wall density in kg/m^3 (Haghpanagh et al. 2013)
    "sorbent_mass": 0.4* 0.08**2 * np.pi / 4 * 0.01 * 55.4,  # Example value for sorbent mass in kg
    "bed_length": 0.01,  # Example value for bed length in meters
    "wall_heat_capacity": 502, # J /kg/K, # Example value for wall heat capacity (Haghpanagh et al. 2013)
    "wall_conductivity" : 16, # K_w, J/m.K.s
    "sorbent_density": 1130, #kg/m3, 55.4 (Haghpanagh et al. 2013)
    "sorbent_heat_capacity": 1070, # J /kg/K, # Example value for solid heat capacity
    "heat_transfer_coefficient": 3,  # Example value for heat transfer coefficient in W/(m^2*K)
    "heat_transfer_coefficient_wall": 26,  # Example value for wall heat transfer coefficient in W/(m^2*K)
    "adsorption_heat_1": -57,  # Example value for adsorption heat of component 1 (CO2) in kJ/mol
    "heat_capacity_1": 30.7 ,  # Example value for adsorbed phase heat capacity of component 1 (CO2) in J/(mol*K) (Haghpanagh et al. 2013)
    "mass_transfer_1": 0.0002,  # Example value for mass transfer coefficient of component 1 (CO2) in s-1
    "adsorption_heat_2": -49,  # Example value for adsorption heat of component 2 (H2O) in kJ/mol
    "heat_capacity_2": 30.7,  # Example value for adsorbed phase heat capacity of component 2 (H2O) in J/(mol*K) (Haghpanagh et al. 2013)
    "mass_transfer_2": 0.002,  # Example value for mass transfer coefficient of component 2 (H2O) in s-1
    "MW_1": 44.01,  # Molecular weight of component 1 (CO2) in g/mol
    "MW_2": 18.02,  # Molecular weight of component 2 (H2O) in g/mol
    "MW_3": 28.02,  # Molecular weight of component 3 (N2) in g/mol
    "MW_4": 32.00,  # Molecular weight of component 4 (O2) in g/mol
    "mu": 1.78e-5,  # Example value for feed viscosity in Pa.s
}

# Define bed inlet values (subject to variation)
inlet_values = {
    "inlet_type": "mass_flow",
    "velocity": 0.1,  # Example value for superficial velocity in m/s
    "rho_gas": 1.13,  # Example value for feed density in kg/m^3
    "gas_thermal_conductivity": 0.09,  # K_z Example value for gas thermal conductivity in W/(m*K*s)
    "feed_mass_flow": (0.1 * float(bed_properties["column_area"]) * 1.13),  # Example value for feed mass flow in kg/s
    "feed_temperature": 298,  # Example value for feed temperature in Kelvin
    "feed_pressure": 101325,  # Example value for feed pressure in Pa
    "mu": 1.78e-5,  # Example value for feed viscosity in Pa.s
    "y1_feed_value": 0.05,  # Example value for feed mole fraction
    "y2_feed_value": 0.01,  # Example value for feed mole fraction
    "y3_feed_value": 0.94,  # Example value for feed mole fraction
}

# Define bed outlet values (subject to variation)
outlet_values = {
    "outlet_type": "pressure",
    "outlet_pressure": 101325,  # Example value for outlet pressure in Pa
    "outlet_temperature": 298,  # Example value for outlet temperature in Kelvin
    "mu": 1.8e-5,  # Example value for outlet viscosity in Pa.s
}

# Initial values for the column
P = np.ones(column_grid["num_cells"]) * 101325  # Example pressure vector in Pa
T = np.ones(column_grid["num_cells"]) * 298  # Example temperature vector in K
Tw = np.ones(column_grid["num_cells"]) * 298  # Example wall temperature vector in K
y1 = np.ones(column_grid["num_cells"]) * 420e-6  # Example mole fraction of component 1
y2 = np.ones(column_grid["num_cells"]) * 100e-6  # Example mole fraction of component 2
y3 = np.ones(column_grid["num_cells"]) * 0.78  # Example mole fraction of component 3
n1 = adsorption_isotherm_1(P, T, y1, y2)[0]  # Example concentration vector for component 1 in mol/m^3
n2 = adsorption_isotherm_2(P, T, y2)[0]  # Example concentration vector for component 2 in mol/m^3
F = np.zeros(8)
E = np.zeros(2)  # Additional variables (e.g., flow rates, mass balances)
initial_conditions = np.concatenate([P, T, Tw, y1, y2, y3, n1, n2, F, E])

# Running simulation! ======================================================================================================

# Implement solver
t_span = [0, 10]  # Time span for the ODE solver
rtol = 1e-6
atol_P = 1e-2 * np.ones(len(P))
atol_T = 1e-4 * np.ones(len(T))
atol_Tw = 1e-4 * np.ones(len(Tw))
atol_y1 = 1e-9 * np.ones(len(y1))
atol_y2 = 1e-9 * np.ones(len(y1))
atol_y3 = 1e-9 * np.ones(len(y1))
atol_n1 = 1e-3 * np.ones(len(n1))
atol_n2 = 1e-3 * np.ones(len(n2))
atol_F = 1e-4 * np.ones(len(F))
atol_E = 1e-4 * np.ones(len(E))
atol_array = np.concatenate([atol_P, atol_T, atol_Tw, atol_y1, atol_y2, atol_y3, atol_n1, atol_n2, atol_F, atol_E])
t0=time.time()
def ODE_func(t, results_vector):
    return ODE_calculations(t, results_vector=results_vector, column_grid=column_grid, bed_properties=bed_properties, inlet_values=inlet_values, outlet_values=outlet_values)
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

print("Mass balance error:", mass_balance_error(F_result, P_result, T_result, y1_result, n1_result, time, bed_properties, column_grid))
print("Energy balance error:", energy_balance_error(E_result, T_result, P_result, y1_result, y2_result, n1_result, n2_result, time, bed_properties, column_grid))


# Create the plot for temperature against time
plt.figure(figsize=(6, 4))
data1 = T_result[0]
data2 = T_result[14]
data3 = T_result[29]

# Plot each dataset
plt.plot(output_matrix.t, data1, label='First node', linewidth=2, marker='o', markersize=3)
plt.plot(output_matrix.t, data2, label='Central node', linewidth=2, marker='s', markersize=3)
plt.plot(output_matrix.t, data3, label='Final node', linewidth=2, marker='^', markersize=3)

# Customize the plot
plt.title('Temperature against time', fontsize=16, fontweight='bold')
plt.xlabel('Time', fontsize=12)

plt.ylabel('Temperature', fontsize=12)
plt.legend(fontsize=11)

plt.grid(True, alpha=0.3)

# Show the plot
plt.show()

# Create the plot for temperature against time
plt.figure(figsize=(6, 4))
data4 = P_result[0]
data5 = P_result[14]
data6 = P_result[29]

# Plot each dataset
plt.plot(output_matrix.t, data4, label='First node', linewidth=2, marker='o', markersize=3)
plt.plot(output_matrix.t, data5, label='Central node', linewidth=2, marker='s', markersize=3)
plt.plot(output_matrix.t, data6, label='Final node', linewidth=2, marker='^', markersize=3)

# Customize the plot
plt.title('Pressure against time', fontsize=16, fontweight='bold')
plt.xlabel('Time', fontsize=12)

plt.ylabel('Pressure', fontsize=12)
plt.legend(fontsize=11)

plt.grid(True, alpha=0.3)

# Show the plot
#plt.show()

# Create the plot for temperature against time
plt.figure(figsize=(6, 4))
data7 = y1_result[0]
data8 = y1_result[14]
data9 = y1_result[29]

# Plot each dataset
plt.plot(output_matrix.t, data7, label='First node', linewidth=2, marker='o', markersize=3)
plt.plot(output_matrix.t, data8, label='Central node', linewidth=2, marker='s', markersize=3)
plt.plot(output_matrix.t, data9, label='Final node', linewidth=2, marker='^', markersize=3)

# Customize the plot
plt.title('Gas phase CO2 against time', fontsize=16, fontweight='bold')
plt.xlabel('Time', fontsize=12)

plt.ylabel('Gas Phase', fontsize=12)
plt.legend(fontsize=11)

plt.grid(True, alpha=0.3)

# Show the plot
#plt.show()

plt.figure(figsize=(6, 4))
data13 = n1_result[0]
data14 = n1_result[14]
data15 = n1_result[29]

# Plot each dataset
plt.plot(output_matrix.t, data13, label='First node', linewidth=2, marker='o', markersize=3)
plt.plot(output_matrix.t, data14, label='Central node', linewidth=2, marker='s', markersize=3)
plt.plot(output_matrix.t, data15, label='Final node', linewidth=2, marker='^', markersize=3)

# Customize the plot
plt.title('CO2 loading against time', fontsize=16, fontweight='bold')
plt.xlabel('Time', fontsize=12)

plt.ylabel('Loading', fontsize=12)
plt.legend(fontsize=11)

plt.grid(True, alpha=0.3)

# Show the plot
plt.show()

# Create the plot for temperature against time
plt.figure(figsize=(6, 4))
data10 = n2_result[0]
data11 = n2_result[14]
data12 = n2_result[29]

# Plot each dataset
plt.plot(output_matrix.t, data10, label='First node', linewidth=2, marker='o', markersize=3)
plt.plot(output_matrix.t, data11, label='Central node', linewidth=2, marker='s', markersize=3)
plt.plot(output_matrix.t, data12, label='Final node', linewidth=2, marker='^', markersize=3)

# Customize the plot
plt.title('CO2 loading against time', fontsize=16, fontweight='bold')
plt.xlabel('Time', fontsize=12)

plt.ylabel('Loading', fontsize=12)
plt.legend(fontsize=11)

plt.grid(True, alpha=0.3)

# Show the plot
plt.show()

# Create the plot for temperature against time
