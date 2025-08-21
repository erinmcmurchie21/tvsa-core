import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution
from additional_functions_validation import create_non_uniform_grid, adsorption_isotherm_1, adsorption_isotherm_2, total_mass_balance_error, CO2_mass_balance_error, energy_balance_error, create_plot, create_combined_plot
import time
import matplotlib.pyplot as plt
import tvsa_adsorption_column_validation as column
import run_adsorption_cycle_validation as run

tot_time = 3000

# Import data
time_observed_temperature = np.loadtxt('pini_data_temperature.csv', delimiter=',', usecols=0)  # First column
temperature_observed = np.loadtxt('pini_data_temperature.csv', delimiter=',', usecols=1)  # Second column
temperature_observed += 273.15  # Adjusting temperature values by adding 273.15

time_observed_molefraction = np.loadtxt('pini_data_molefraction.csv', delimiter=',', usecols=0)  # First column
molefraction_observed = np.loadtxt('pini_data_molefraction.csv', delimiter=',', usecols=1)  # Second column

observed_data = {
    "temperature": temperature_observed,
    "t_temperature": time_observed_temperature,
    "molefraction": molefraction_observed,
    "t_molefraction": time_observed_molefraction
}

# Set up simulation conditions
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

optimization_history = []

best_params = {
             "h_bed": 40,
             "h_wall": 10,
        }

def run_best_simulation(simulation_conditions, best_params):
        """
        Run simulation with best fitted parameters.
        
        Returns:
        --------
        tuple
            (output_matrix, bed_properties_fitted)
        """
          # Example fitted parameters, replace with actual fitted values

        if best_params is None:
            raise ValueError("No fitted parameters available. Run fit_parameters first.")

        # Update bed properties with best parameters
        bed_properties = simulation_conditions["bed_properties"].copy()
        bed_properties.update(best_params)
        
        # Set up simulation
        column_grid = simulation_conditions["column_grid"]
        initial_conditions = simulation_conditions["initial_conditions"]
        inlet_values = simulation_conditions["inlet_values"]
        outlet_values = simulation_conditions["outlet_values"]
        rtol = simulation_conditions["rtol"]
        atol_array = simulation_conditions["atol_array"]
        
        
        time_span = (0, tot_time)
        
        # Define ODE function
        def ODE_func(t, results_vector):
            return column.ODE_calculations(t, results_vector=results_vector, 
                                         column_grid=column_grid, 
                                         bed_properties=bed_properties, 
                                         inlet_values=inlet_values, 
                                         outlet_values=outlet_values)
        
        # Solve ODE
        output_matrix = solve_ivp(ODE_func, time_span, initial_conditions, 
                                method='BDF', 
                                rtol=rtol, atol=atol_array)
        
        P_walls_result, T_walls_result, Tw_walls_result,\
            y1_walls_result, y2_walls_result, y3_walls_result, \
            v_walls_result = column.final_wall_values(column_grid, bed_properties, inlet_values, outlet_values, output_matrix)
        
        return output_matrix, y1_walls_result, bed_properties


def run_initial_simulation(simulation_conditions):

        bed_properties = simulation_conditions["bed_properties"]
        # Set up simulation
        column_grid = simulation_conditions["column_grid"]
        initial_conditions = simulation_conditions["initial_conditions"]
        inlet_values = simulation_conditions["inlet_values"]
        outlet_values = simulation_conditions["outlet_values"]
        rtol = simulation_conditions["rtol"]
        atol_array = simulation_conditions["atol_array"]
    
        # Short time span for testing
        time_span = (0, tot_time)  # Just 100 seconds
        
        def ODE_func(t, results_vector):
            return column.ODE_calculations(
                t, 
                results_vector=results_vector, 
                column_grid=simulation_conditions["column_grid"], 
                bed_properties=simulation_conditions["bed_properties"], 
                inlet_values=simulation_conditions["inlet_values"], 
                outlet_values=simulation_conditions["outlet_values"]
            )

        print("Running short simulation (3000 seconds)...")
        output_matrix = solve_ivp(
            ODE_func, 
            time_span, 
            simulation_conditions["initial_conditions"], 
            method='BDF', 
            rtol=simulation_conditions["rtol"], 
            atol=simulation_conditions["atol_array"]
        )

        P_walls_result, T_walls_result, Tw_walls_result,\
            y1_walls_result, y2_walls_result, y3_walls_result, \
            v_walls_result = column.final_wall_values(column_grid, bed_properties, inlet_values, outlet_values, output_matrix)

        return output_matrix, y1_walls_result, bed_properties

def objective_function(best_params, simulation_conditions, observed_data):

    # Create a copy of bed_properties to avoid modifying the original
    bed_properties = simulation_conditions["bed_properties"].copy()

    bed_properties["h_bed"] = best_params["h_bed"]
    bed_properties["h_wall"] = best_params["h_wall"]

    # Set up simulation
    column_grid = simulation_conditions["column_grid"]
    initial_conditions = simulation_conditions["initial_conditions"]
    inlet_values = simulation_conditions["inlet_values"]
    outlet_values = simulation_conditions["outlet_values"]
    rtol = simulation_conditions["rtol"]
    atol_array = simulation_conditions["atol_array"]
    
    # Determine time span for ODE solver based on data
    time_span = (0, tot_time)
    
    # Define ODE function
    def ODE_func(t, results_vector):
        return column.ODE_calculations(t, results_vector=results_vector, 
                                        column_grid=column_grid, 
                                        bed_properties=bed_properties, 
                                        inlet_values=inlet_values, 
                                        outlet_values=outlet_values)
    
    # Solve ODE
    output_matrix = solve_ivp(ODE_func, time_span, initial_conditions, 
                                method='BDF', rtol=rtol, atol=atol_array)

    P_walls_result, T_walls_result, Tw_walls_result,\
        y1_walls_result, y2_walls_result, y3_walls_result, \
        v_walls_result = column.final_wall_values(column_grid, bed_properties, inlet_values, outlet_values, output_matrix)
    
    # Extract simulation results
    T_sim = output_matrix.y[2*column_grid["num_cells"]:3*column_grid["num_cells"]] * bed_properties["T_ref"]
    y1_sim = y1_walls_result[-1,:]

    # Calculate errors
    total_error = 0
    error_components = {}
    
    # Use provided weights or defaults
    weights = {'temperature': 1.0, 'molefraction': 1.0}

    # Temperature error - Use keys that match test_fitting_function.py
    T_observed = observed_data["temperature"]
    time_temp_observed = observed_data["t_temperature"]

    # Interpolate simulation to experimental time points
    temp_sim_interp = np.interp(time_temp_observed, output_matrix.t, T_sim[9, :])
        #                            ↑            ↑                ↑
        #                    experimental    simulation      outlet node
        #                    time points     time points     temperature


    temp_error = np.abs(100 / len(T_observed) *np.sum((temp_sim_interp - T_observed)/ T_observed))
    error_components['temperature'] = temp_error
    total_error += weights.get('temperature', 1.0) * temp_error

    # Mole fraction error - Use keys that match test_fitting_function.py
    y1_observed = observed_data["molefraction"]
    t_molefraction_observed = observed_data["t_molefraction"]

    # Interpolate simulation to experimental time points
    y1_sim_interp = np.interp(t_molefraction_observed, output_matrix.t, y1_sim)

    molefraction_error = np.abs(100 / len(y1_observed) *np.sum((y1_sim_interp - y1_observed)/ y1_observed))
    error_components['molefraction'] = molefraction_error
    total_error += weights.get('molefraction', 1.0) * molefraction_error


    print(f"Error = {total_error:.6e}")

    return total_error

def plot_results(simulation_conditions, observed_data, best_params):

    output_matrix_fitted, y1_walls_result_fitted, bed_properties = run_best_simulation(simulation_conditions, best_params)
    #output_matrix_initial, y1_walls_result_initial, bed_properties_initial = run_initial_simulation(simulation_conditions)
    column_grid = simulation_conditions["column_grid"]
    # Extract simulation results
    #T_sim_initial = output_matrix_initial.y[2*column_grid["num_cells"]:3*column_grid["num_cells"]] * bed_properties_initial["T_ref"]
    #y1_sim_initial = y1_walls_result_initial
    T_sim_fitted = output_matrix_fitted.y[2*column_grid["num_cells"]:3*column_grid["num_cells"]] * bed_properties["T_ref"]
    y1_sim_fitted = y1_walls_result_fitted

    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Temperature plot
    axes[0].plot(output_matrix_fitted.t, T_sim_fitted[9, :], 
                        label="Fitted (h_bed = 140, h_wall = 20)", linewidth=2, linestyle='-')
    axes[0].plot(observed_data['t_temperature'], observed_data['temperature'], 
                'ro', label='Experimental', markersize=4)
    #axes[0].plot(output_matrix_initial.t, T_sim_initial[-1, :], 
                        #label="Initial (h_bed = 50, h_wall = 7)", linewidth=2, linestyle='-')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Temperature (K)')
    axes[0].set_title('Temperature Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Mole fraction plot
    axes[1].plot(output_matrix_fitted.t, y1_sim_fitted[-1,:], 
                        label="Fitted (h_bed = 140, h_wall = 20)", linewidth=2, linestyle='-')
    axes[1].plot(observed_data['t_molefraction'], observed_data['molefraction'], 
                'ro', label='Experimental', markersize=4)
    #axes[1].plot(output_matrix_initial.t, y1_sim_initial[-1,:], 
                        #label="Initial (h_bed = 50, h_wall = 7)", linewidth=2, linestyle='-')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Mole Fraction')
    axes[1].set_title('Mole Fraction Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)


    plt.tight_layout()
    plt.show()
    
    # Print parameter values
    print("\nFitted Parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value:.4f}")

def main(simulation_conditions, observed_data, best_params):

    objective_function(best_params, simulation_conditions, observed_data)
    
    plot_results(simulation_conditions, observed_data, best_params)


if __name__ == "__main__":
    main(simulation_conditions, observed_data, best_params)

