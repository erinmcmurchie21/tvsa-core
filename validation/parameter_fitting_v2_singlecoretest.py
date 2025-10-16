import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
import time
import matplotlib.pyplot as plt
import tvsa_adsorption_column_validation as column
import run_adsorption_cycle_validation as run

# Create a function to perform parameter fitting


class ParameterFitter:
    """
    A class to handle parameter fitting for the adsorption column model.
    """

    def __init__(self, simulation_conditions, observed_data):
        """
        Initialize the parameter fitter.

        Parameters:
        -----------
        simulation_conditions : dict
            Contains all simulation setup (bed_properties, grid, etc.)
        observed_data : dict
            Contains experimental data with keys like 't_temperature', 'temperature',
            't_molefraction', 'molefraction', etc.
        """
        self.simulation_conditions = simulation_conditions
        self.observed_data = observed_data
        self.best_params = None
        self.best_result = None
        self.optimization_history = []
        self.tot_time = 3000

    """# Define the objective function for parameter fitting
    def objective_function(self, parameters, param_names, weights=None):
        
        # Create a copy of bed_properties to avoid modifying the original
        bed_properties = self.simulation_conditions["bed_properties"].copy()
        
        # Update parameters
        for i, param_name in enumerate(param_names):
            bed_properties[param_name] = parameters[i]

        # Set up simulation
        column_grid = self.simulation_conditions["column_grid"]
        initial_conditions = self.simulation_conditions["initial_conditions"]
        left_values = self.simulation_conditions["left_values"]
        right_values = self.simulation_conditions["right_values"]
        column_direction = self.simulation_conditions["column_direction"]
        rtol = self.simulation_conditions["rtol"]
        atol_array = self.simulation_conditions["atol_array"]
        
        # Determine time span for ODE solver based on data
        time_span = (0, self.tot_time)
        
        # Define ODE function
        def ODE_func(t, results_vector):
            return column.ODE_calculations(t, results_vector=results_vector, 
                                            column_grid=column_grid, 
                                            bed_properties=bed_properties, 
                                            left_values=left_values, 
                                            right_values=right_values, 
                                            column_direction=column_direction)

        # Solve ODE
        output_matrix = solve_ivp(ODE_func, time_span, initial_conditions, 
                                    method='BDF', rtol=rtol, atol=atol_array)

        P_walls_result, T_walls_result, Tw_walls_result,\
            y1_walls_result, y2_walls_result, y3_walls_result, \
            v_walls_result = column.final_wall_values(column_grid, bed_properties, left_values, right_values, output_matrix)
        # Extract simulation results
        T_sim = output_matrix.y[column_grid["num_cells"]:2*column_grid["num_cells"]] * bed_properties["T_ref"]
        y1_sim = y1_walls_result[-1,:]

        # Calculate errors
        total_error = 0
        error_components = {}
        
        # Use provided weights or defaults
        if weights is None:
            weights = {'temperature': 1.0, 'molefraction': 1.0}

        # Temperature error - Use keys that match test_fitting_function.py
        T_observed = self.observed_data["temperature"]
        time_temp_observed = self.observed_data["t_temperature"]

        # Interpolate simulation to experimental time points
        temp_sim_interp = np.interp(time_temp_observed, output_matrix.t, T_sim[9, :])
            #                            ↑            ↑                ↑
            #                    experimental    simulation      outlet node
            #                    time points     time points     temperature


        temp_error = np.abs(100 / len(T_observed) *np.sum((temp_sim_interp - T_observed)/ T_observed))
        error_components['temperature'] = temp_error
        total_error += weights.get('temperature', 1.0) * temp_error

        # Mole fraction error - Use keys that match test_fitting_function.py
        y1_observed = self.observed_data["molefraction"]
        t_molefraction_observed = self.observed_data["t_molefraction"]

        # Interpolate simulation to experimental time points
        y1_sim_interp = np.interp(t_molefraction_observed, output_matrix.t, y1_sim)

        molefraction_error = np.abs(100 / len(y1_observed) *np.sum((y1_sim_interp - y1_observed)/ y1_observed))
        error_components['molefraction'] = molefraction_error
        total_error += weights.get('molefraction', 1.0) * molefraction_error

        self.optimization_history.append({
                'params': parameters.copy(),
                'total_error': total_error,
                'error_components': error_components.copy()
            })
        
        # Print progress
        if len(self.optimization_history) % 1 == 0:
            print(f"Iteration {len(self.optimization_history)}: Error = {total_error:.6e}, Parameters = {parameters}")

        return total_error
      """

    def minimiser(self, param_names, initial_guess, bounds, weights=None):
        # FIX: Remove 'parameters' from signature - it's not needed
        global global_param_names, global_initial_guess, global_bounds, global_weights

        global_param_names = param_names
        global_initial_guess = initial_guess
        global_bounds = bounds
        global_weights = weights

        print(f"Starting parameter fitting for: {param_names}")
        print(f"Initial guess: {initial_guess}")
        print(f"Bounds: {bounds}")
        start_time = time.time()

        # Clear optimization history
        self.optimization_history = []

        # Perform the optimization using the global function to avoid pickling issues
        result = differential_evolution(
            objective_for_DE,
            bounds,
            seed=42,
            maxiter=0,
            popsize=5,
            atol=1e-6,
            tol=1e-6,
            disp=True,
            init="sobol",
            polish=False,
            # workers=1,  # Enable parallel processing
            # updating= 'immediate'  # Required for parallel processing
        )

        end_time = time.time()

        print(f"\nOptimization completed in {end_time - start_time:.2f} seconds")
        print(f"Success: {result.success}")
        print(f"Final parameters: {dict(zip(param_names, result.x))}")
        print(f"Final error: {result.fun}")

        self.best_params = dict(zip(param_names, result.x))
        self.best_result = result

        return result

        # Run simulation one more time with optimized parameters to get final output_matrix

    def run_best_simulation(self):
        """
        Run simulation with best fitted parameters.

        Returns:
        --------
        tuple
            (output_matrix, bed_properties_fitted)
        """
        if self.best_params is None:
            raise ValueError(
                "No fitted parameters available. Run fit_parameters first."
            )

        # Update bed properties with best parameters
        bed_properties = self.simulation_conditions["bed_properties"].copy()
        bed_properties.update(self.best_params)

        # Set up simulation
        column_grid = self.simulation_conditions["column_grid"]
        initial_conditions = self.simulation_conditions["initial_conditions"]
        left_values = self.simulation_conditions["left_values"]
        right_values = self.simulation_conditions["right_values"]
        column_direction = self.simulation_conditions["column_direction"]
        rtol = self.simulation_conditions["rtol"]
        atol_array = self.simulation_conditions["atol_array"]

        time_span = (0, self.tot_time)

        # Define ODE function
        def ODE_func(t, results_vector):
            return column.ODE_calculations(
                t,
                results_vector=results_vector,
                column_grid=column_grid,
                bed_properties=bed_properties,
                left_values=left_values,
                right_values=right_values,
                column_direction=column_direction,
            )

        # Solve ODE
        output_matrix = solve_ivp(
            ODE_func,
            time_span,
            initial_conditions,
            method="BDF",
            rtol=rtol,
            atol=atol_array,
        )

        (
            P_walls_result,
            T_walls_result,
            Tw_walls_result,
            y1_walls_result,
            y2_walls_result,
            y3_walls_result,
            v_walls_result,
        ) = column.final_wall_values(
            column_grid, bed_properties, left_values, right_values, output_matrix
        )

        return output_matrix, y1_walls_result, bed_properties

    def plot_results(self):
        output_matrix_fitted, y1_walls_result_fitted, bed_properties = (
            self.run_best_simulation()
        )
        # output_matrix_initial, y1_walls_result_initial, bed_properties_initial = self.run_initial_simulation()
        column_grid = self.simulation_conditions["column_grid"]
        # Extract simulation results
        # T_sim_initial = output_matrix_initial.y[2*column_grid["num_cells"]:3*column_grid["num_cells"]]
        # y1_sim_initial = y1_walls_result_initia
        T_sim_fitted = (
            output_matrix_fitted.y[
                column_grid["num_cells"] : 2 * column_grid["num_cells"]
            ]
            * bed_properties["T_ref"]
        )
        y1_sim_fitted = y1_walls_result_fitted

        # Create plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # Temperature plot
        axes[0].plot(
            output_matrix_fitted.t,
            T_sim_fitted[9, :],
            label="Fitted",
            linewidth=2,
            linestyle="-",
        )
        axes[0].plot(
            self.observed_data["t_temperature"],
            self.observed_data["temperature"],
            "ro",
            label="Experimental",
            markersize=4,
        )
        # axes[0].plot(output_matrix_initial.t, T_sim_initial[9, :],
        # label="Initial", linewidth=2, linestyle='-')
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Temperature (K)")
        axes[0].set_title("Temperature Comparison")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Mole fraction plot
        axes[1].plot(
            output_matrix_fitted.t,
            y1_sim_fitted[-2, :],
            label="Fitted",
            linewidth=2,
            linestyle="-",
        )
        axes[1].plot(
            self.observed_data["t_molefraction"],
            self.observed_data["molefraction"],
            "ro",
            label="Experimental",
            markersize=4,
        )
        # axes[1].plot(output_matrix_initial.t, y1_sim_initial[-2,:],
        # label="Initial", linewidth=2, linestyle='-')
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Mole Fraction")
        axes[1].set_title("Mole Fraction Comparison")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print parameter values
        print("\nFitted Parameters:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value:.4f}")


def global_objective_function(
    parameters,
    initial_guess,
    simulation_conditions,
    observed_data,
    param_names,
    weights,
    tot_time,
    optimization_history,
):
    # Create a copy of bed_properties to avoid modifying the original
    bed_properties = simulation_conditions["bed_properties"].copy()

    # Update parameters
    for i, param_name in enumerate(param_names):
        bed_properties[param_name] = parameters[i]

    # Set up simulation
    column_grid = simulation_conditions["column_grid"]
    initial_conditions = simulation_conditions["initial_conditions"]
    left_values = simulation_conditions["left_values"]
    right_values = simulation_conditions["right_values"]
    column_direction = simulation_conditions["column_direction"]
    rtol = simulation_conditions["rtol"]
    atol_array = simulation_conditions["atol_array"]

    # Determine time span for ODE solver based on data
    time_span = (0, tot_time)

    # Define ODE function
    def ODE_func(t, results_vector):
        return column.ODE_calculations(
            t,
            results_vector=results_vector,
            column_grid=column_grid,
            bed_properties=bed_properties,
            left_values=left_values,
            right_values=right_values,
            column_direction=column_direction,
        )

    # Solve ODE
    output_matrix = solve_ivp(
        ODE_func,
        time_span,
        initial_conditions,
        method="BDF",
        rtol=rtol,
        atol=atol_array,
    )

    (
        P_walls_result,
        T_walls_result,
        Tw_walls_result,
        y1_walls_result,
        y2_walls_result,
        y3_walls_result,
        v_walls_result,
    ) = column.final_wall_values(
        column_grid, bed_properties, left_values, right_values, output_matrix
    )
    # Extract simulation results
    T_sim = (
        output_matrix.y[column_grid["num_cells"] : 2 * column_grid["num_cells"]]
        * bed_properties["T_ref"]
    )
    y1_sim = y1_walls_result[-1, :]

    # Calculate errors
    total_error = 0
    error_components = {}

    # Use provided weights or defaults
    if weights is None:
        weights = {"temperature": 1.0, "molefraction": 1.0}

    # Temperature error - Use keys that match test_fitting_function.py
    T_observed = observed_data["temperature"]
    time_temp_observed = observed_data["t_temperature"]

    # Interpolate simulation to experimental time points
    temp_sim_interp = np.interp(time_temp_observed, output_matrix.t, T_sim[9, :])
    #                            ↑            ↑                ↑
    #                    experimental    simulation      outlet node
    #                    time points     time points     temperature

    temp_error = np.abs(
        100 / len(T_observed) * np.sum((temp_sim_interp - T_observed) / T_observed)
    )
    error_components["temperature"] = temp_error
    total_error += weights.get("temperature", 1.0) * temp_error

    # Mole fraction error - Use keys that match test_fitting_function.py
    y1_observed = observed_data["molefraction"]
    t_molefraction_observed = observed_data["t_molefraction"]

    # Interpolate simulation to experimental time points
    y1_sim_interp = np.interp(t_molefraction_observed, output_matrix.t, y1_sim)

    molefraction_error = np.abs(
        100 / len(y1_observed) * np.sum((y1_sim_interp - y1_observed) / y1_observed)
    )
    error_components["molefraction"] = molefraction_error
    total_error += weights.get("molefraction", 1.0) * molefraction_error

    optimization_history.append(
        {
            "params": parameters.copy(),
            "total_error": total_error,
            "error_components": error_components.copy(),
        }
    )

    # Print progress
    if len(optimization_history) % 1 == 0:
        print(
            f"Iteration {len(optimization_history)}: Error = {total_error:.6e}, Parameters = {parameters}"
        )

    return total_error


def objective_for_DE(params):
    global \
        global_param_names, \
        global_simulation_conditions, \
        global_observed_data, \
        global_weights, \
        global_tot_time, \
        global_optimization_history, \
        global_initial_guess, \
        global_bounds
    return global_objective_function(
        params,
        global_initial_guess,
        global_simulation_conditions,
        global_observed_data,
        global_param_names,
        global_weights,
        global_tot_time,
        global_optimization_history,
    )


# function to perform parameter fitting


def main():
    global \
        global_param_names, \
        global_simulation_conditions, \
        global_observed_data, \
        global_weights, \
        global_tot_time, \
        global_optimization_history, \
        global_initial_guess, \
        global_bounds

    # Import data
    time_observed_temperature = np.loadtxt(
        "data/pini_data_temperature.csv", delimiter=",", usecols=0
    )  # First column
    temperature_observed = np.loadtxt(
        "data/pini_data_temperature.csv", delimiter=",", usecols=1
    )  # Second column
    temperature_observed += 273.15  # Adjusting temperature values by adding 273.15

    time_observed_molefraction = np.loadtxt(
        "data/pini_data_molefraction.csv", delimiter=",", usecols=0
    )  # First column
    molefraction_observed = np.loadtxt(
        "data/pini_data_molefraction.csv", delimiter=",", usecols=1
    )  # Second column

    observed_data = {
        "temperature": temperature_observed,
        "t_temperature": time_observed_temperature,
        "molefraction": molefraction_observed,
        "t_molefraction": time_observed_molefraction,
    }

    # Set up simulation conditions
    bed_properties, column_grid, initial_conditions, rtol, atol_array = (
        run.create_fixed_properties()
    )
    left_values, right_values, column_direction = run.define_boundary_conditions(
        bed_properties
    )

    simulation_conditions = {
        "bed_properties": bed_properties,
        "column_grid": column_grid,
        "initial_conditions": initial_conditions,
        "left_values": left_values,
        "right_values": right_values,
        "column_direction": column_direction,
        "rtol": rtol,
        "atol_array": atol_array,
    }

    # Set global variables for multiprocessing
    global_simulation_conditions = simulation_conditions
    global_observed_data = observed_data
    global_tot_time = 3000
    global_optimization_history = []

    # Create parameter fitter
    fitter = ParameterFitter(simulation_conditions, observed_data)

    # Define parameters to fit
    param_names = ["h_bed", "h_wall", "K_z"]
    initial_guess = [76.0, 40.0, 0.4]
    bounds = [(50, 100), (25, 75), (0, 1)]  # (min, max) for each parameter

    weights = {"temperature": 1.0, "molefraction": 1.0}

    # Set remaining global variables
    global_param_names = param_names
    global_initial_guess = initial_guess
    global_bounds = bounds
    global_weights = weights

    # Fit parameters
    result = fitter.minimiser(
        param_names=param_names,
        initial_guess=initial_guess,
        bounds=bounds,
        weights=weights,
    )

    output_matrix_fitted, y1_walls_result_fitted, bed_properties = (
        fitter.run_best_simulation()
    )
    T_sim_fitted = (
        output_matrix_fitted.y[column_grid["num_cells"] : 2 * column_grid["num_cells"]]
        * bed_properties["T_ref"]
    )
    y1_sim_fitted = y1_walls_result_fitted

    # Output data
    print("\nFitted Parameters:")
    print(f"Best parameters: {fitter.best_params}")
    print(f"Final error: {result.fun}")
    print(f"T_sim_fitted: {T_sim_fitted}")
    print(f"y1_sim_fitted: {y1_sim_fitted}")

    # Plot results
    # fitter.plot_results()


if __name__ == "__main__":
    main()
