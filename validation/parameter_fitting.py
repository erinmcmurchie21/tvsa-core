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

    # Define the objective function for parameter fitting
    def objective_function(self, parameters, param_names, weights=None):
        # Create a copy of bed_properties to avoid modifying the original
        bed_properties = self.simulation_conditions["bed_properties"].copy()

        # Update parameters
        for i, param_name in enumerate(param_names):
            bed_properties[param_name] = parameters[i]

        # Set up simulation
        column_grid = self.simulation_conditions["column_grid"]
        initial_conditions = self.simulation_conditions["initial_conditions"]
        inlet_values = self.simulation_conditions["inlet_values"]
        outlet_values = self.simulation_conditions["outlet_values"]
        rtol = self.simulation_conditions["rtol"]
        atol_array = self.simulation_conditions["atol_array"]

        # Determine time span for ODE solver based on data
        time_span = (0, self.tot_time)

        # Define ODE function
        def ODE_func(t, results_vector):
            return column.ODE_calculations(
                t,
                results_vector=results_vector,
                column_grid=column_grid,
                bed_properties=bed_properties,
                inlet_values=inlet_values,
                outlet_values=outlet_values,
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
            column_grid, bed_properties, inlet_values, outlet_values, output_matrix
        )
        # Extract simulation results
        T_sim = (
            output_matrix.y[2 * column_grid["num_cells"] : 3 * column_grid["num_cells"]]
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
        T_observed = self.observed_data["temperature"]
        time_temp_observed = self.observed_data["t_temperature"]

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
        y1_observed = self.observed_data["molefraction"]
        t_molefraction_observed = self.observed_data["t_molefraction"]

        # Interpolate simulation to experimental time points
        y1_sim_interp = np.interp(t_molefraction_observed, output_matrix.t, y1_sim)

        molefraction_error = np.abs(
            100 / len(y1_observed) * np.sum((y1_sim_interp - y1_observed) / y1_observed)
        )
        error_components["molefraction"] = molefraction_error
        total_error += weights.get("molefraction", 1.0) * molefraction_error

        self.optimization_history.append(
            {
                "params": parameters.copy(),
                "total_error": total_error,
                "error_components": error_components.copy(),
            }
        )

        # Print progress
        if len(self.optimization_history) % 1 == 0:
            print(
                f"Iteration {len(self.optimization_history)}: Error = {total_error:.6e}, Parameters = {parameters}"
            )

        return total_error

    def minimiser(self, param_names, initial_guess, bounds, weights=None):
        # FIX: Remove 'parameters' from signature - it's not needed

        print(f"Starting parameter fitting for: {param_names}")
        print(f"Initial guess: {initial_guess}")
        print(f"Bounds: {bounds}")
        start_time = time.time()

        # Clear optimization history
        self.optimization_history = []

        # Perform the optimization
        result = differential_evolution(
            lambda params: self.objective_function(params, param_names, weights),
            bounds,
            seed=42,
            maxiter=0,
            popsize=1,
            atol=1e-6,
            tol=1e-6,
            disp=True,
            init="sobol",
            polish=False,
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
        inlet_values = self.simulation_conditions["inlet_values"]
        outlet_values = self.simulation_conditions["outlet_values"]
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
                inlet_values=inlet_values,
                outlet_values=outlet_values,
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
            column_grid, bed_properties, inlet_values, outlet_values, output_matrix
        )

        return output_matrix, y1_walls_result, bed_properties

    def run_initial_simulation(self):
        bed_properties = self.simulation_conditions["bed_properties"]
        # Set up simulation
        column_grid = self.simulation_conditions["column_grid"]
        initial_conditions = self.simulation_conditions["initial_conditions"]
        inlet_values = self.simulation_conditions["inlet_values"]
        outlet_values = self.simulation_conditions["outlet_values"]
        rtol = self.simulation_conditions["rtol"]
        atol_array = self.simulation_conditions["atol_array"]

        # Short time span for testing
        time_span = (0, self.tot_time)  # Just 100 seconds

        def ODE_func(t, results_vector):
            return column.ODE_calculations(
                t,
                results_vector=results_vector,
                column_grid=self.simulation_conditions["column_grid"],
                bed_properties=self.simulation_conditions["bed_properties"],
                inlet_values=self.simulation_conditions["inlet_values"],
                outlet_values=self.simulation_conditions["outlet_values"],
            )

        print("Running short simulation (3000 seconds)...")
        output_matrix = solve_ivp(
            ODE_func,
            time_span,
            self.simulation_conditions["initial_conditions"],
            method="BDF",
            rtol=self.simulation_conditions["rtol"],
            atol=self.simulation_conditions["atol_array"],
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
            column_grid, bed_properties, inlet_values, outlet_values, output_matrix
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
        # y1_sim_initial = y1_walls_result_initial
        T_sim_fitted = (
            output_matrix_fitted.y[
                2 * column_grid["num_cells"] : 3 * column_grid["num_cells"]
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


# function to perform parameter fitting


def main(self):
    # Import data
    time_observed_temperature = np.loadtxt(
        "pini_data_temperature.csv", delimiter=",", usecols=0
    )  # First column
    temperature_observed = (
        np.loadtxt("pini_data_temperature.csv", delimiter=",", usecols=1) + 273.15
    )  # Second column

    time_observed_molefraction = np.loadtxt(
        "pini_data_molefraction.csv", delimiter=",", usecols=0
    )  # First column
    molefraction_observed = np.loadtxt(
        "pini_data_molefraction.csv", delimiter=",", usecols=1
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
    inlet_values, outlet_values = run.define_boundary_conditions(bed_properties)

    simulation_conditions = {
        "bed_properties": bed_properties,
        "column_grid": column_grid,
        "initial_conditions": initial_conditions,
        "inlet_values": inlet_values,
        "outlet_values": outlet_values,
        "rtol": rtol,
        "atol_array": atol_array,
    }

    # Create parameter fitter
    fitter = ParameterFitter(simulation_conditions, observed_data)

    # Define parameters to fit
    param_names = ["h_bed", "h_wall"]
    initial_guess = [30.0, 10.0]
    bounds = [(5, 50), (5, 50)]  # (min, max) for each parameter

    weights = {"temperature": 1.0, "molefraction": 1.0}
    # Fit parameters
    result = fitter.minimiser(
        param_names=param_names,
        initial_guess=initial_guess,
        bounds=bounds,
        weights=weights,
    )

    output_matrix_fitted, y1_walls_result_fitted, bed_properties = (
        self.run_best_simulation()
    )
    T_sim_fitted = (
        output_matrix_fitted.y[
            2 * column_grid["num_cells"] : 3 * column_grid["num_cells"]
        ]
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
