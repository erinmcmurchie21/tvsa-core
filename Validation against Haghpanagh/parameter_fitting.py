import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from additional_functions_validation import create_non_uniform_grid, adsorption_isotherm_1, adsorption_isotherm_2, total_mass_balance_error, CO2_mass_balance_error, energy_balance_error, create_plot, create_combined_plot
import time
import matplotlib.pyplot as plt
import tvsa_adsorption_column_validation as column
import run_adsorption_cycle_validation as run

# Import experimental data
# Option 1: If your CSV has comma delimiters
t_observed = np.loadtxt('pini_data.csv', delimiter=',', usecols=0)  # First column
temperature_observed = np.loadtxt('pini_data.csv', delimiter=',', usecols=1)  # Second column
temperature_observed = temperature_observed + 273.15


bed_properties, column_grid, initial_conditions, rtol, atol_array = run.create_fixed_properties()
inlet_values, outlet_values = run.define_boundary_conditions(bed_properties)


simulation_conditions = {
    "bed_properties": bed_properties,
    "column_grid": column_grid,
    "initial_conditions": initial_conditions,
    "inlet_values": inlet_values,
    "outlet_values": outlet_values,
    "rtol": rtol,
    "atol_array": atol_array
}

# Create a function to perform parameter fitting

def parameter_fitting(parameters, simulation_conditions, T_observed, t_observed):
    # Extract variables from simulation_conditions
    column_grid = simulation_conditions["column_grid"]
    original_bed_properties = simulation_conditions["bed_properties"]
    initial_conditions = simulation_conditions["initial_conditions"]
    inlet_values = simulation_conditions["inlet_values"]
    outlet_values = simulation_conditions["outlet_values"]
    rtol = simulation_conditions["rtol"]
    atol_array = simulation_conditions["atol_array"]

    # Define the objective function for parameter fitting
    def objective_function(params):
        # Create a copy of bed_properties to avoid modifying the original
        bed_properties = original_bed_properties.copy()
        
        # Update parameters
        bed_properties["h_bed"], bed_properties["h_wall"] = params
        
        # Solve the system of ODEs
        time_span = (0, t_observed[-1]) # Use the last time point from the observed data
        
        t0=time.time()
        def ODE_func(t, results_vector):
            return column.ODE_calculations(t, results_vector=results_vector, column_grid=column_grid, bed_properties=bed_properties, inlet_values=inlet_values, outlet_values=outlet_values)
        
        output_matrix = solve_ivp(ODE_func, time_span, initial_conditions, method='BDF', t_eval=t_observed, rtol=rtol, atol=atol_array)
        t1=time.time()
        total_time = t1 - t0

        T_result = output_matrix.y[column_grid["num_cells"]:2*column_grid["num_cells"]]
        # Calculate the errors
        least_squares_error = np.sum((T_result - T_observed)**2)
        # Return the total error
        return least_squares_error

    # Perform the optimization
    result = minimize(objective_function, parameters, method='Nelder-Mead')

    # Run simulation one more time with optimized parameters to get final output_matrix
    final_bed_properties = original_bed_properties.copy()
    final_bed_properties["h_bed"], final_bed_properties["h_wall"] = result.x
    time_span = (0, t_observed[-1])
    def ODE_func(t, results_vector):
        return column.ODE_calculations(t, results_vector=results_vector, column_grid=column_grid, bed_properties=final_bed_properties, inlet_values=inlet_values, outlet_values=outlet_values)
    output_matrix = solve_ivp(ODE_func, time_span, initial_conditions, method='BDF', t_eval=t_observed, rtol=rtol, atol=atol_array)

    # Plot the results
    create_plot(output_matrix, column_grid)

    return result, output_matrix, final_bed_properties

initial_guess = [20.0, 140.0]
bounds = [(1, 100), (1, 200)]

output = parameter_fitting(initial_guess, simulation_conditions, temperature_observed, t_observed)
output_matrix = output[1]
time = output_matrix.t
T_fitted = output_matrix.y[column_grid["num_cells"]:2*column_grid["num_cells"]]
print(f"Fitted heat transfer coefficients: {output[0].x}")

plt.figure(figsize=(6, 4))
plt.plot(time, T_fitted, label="Fitted", linewidth=2, marker='o', markersize=3)
plt.plot(t_observed, temperature_observed, label="Observed", linestyle='--', linewidth=2, marker='x', markersize=3)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Temperature / K', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.show()