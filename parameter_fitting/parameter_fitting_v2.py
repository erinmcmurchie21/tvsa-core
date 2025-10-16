import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import tvsa_adsorption_column_validation as column
import run_adsorption_cycle_validation as run


def import_data():
    time_observed_temperature = np.loadtxt(
        "data/pini_data_temperature_2.csv", delimiter=",", usecols=0
    )
    temperature_observed = np.loadtxt(
        "data/pini_data_temperature_2.csv", delimiter=",", usecols=1
    )
    temperature_observed += 273.15

    time_observed_molefraction = np.loadtxt(
        "data/pini_data_molefraction_2.csv", delimiter=",", usecols=0
    )
    molefraction_observed = np.loadtxt(
        "data/pini_data_molefraction_2.csv", delimiter=",", usecols=1
    )

    observed_data = {
        "temperature": temperature_observed,
        "t_temperature": time_observed_temperature,
        "molefraction": molefraction_observed,
        "t_molefraction": time_observed_molefraction,
    }
    return observed_data


def objective_function(
    parameters,
):
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
    param_names = ["h_bed", "h_wall", "K_z"]
    initial_guess = [100, 60.0, 0.4]
    bounds = [(50, 150), (25, 75), (0, 1)]
    observed_data = import_data()
    bed_properties = simulation_conditions["bed_properties"].copy()
    for i, param_name in enumerate(param_names):
        bed_properties[param_name] = parameters[i]

    column_grid = simulation_conditions["column_grid"]
    initial_conditions = simulation_conditions["initial_conditions"]
    left_values = simulation_conditions["left_values"]
    right_values = simulation_conditions["right_values"]
    column_direction = simulation_conditions["column_direction"]
    rtol = simulation_conditions["rtol"]
    atol_array = simulation_conditions["atol_array"]
    time_span = (0, 3000)

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

    output_matrix = solve_ivp(
        ODE_func,
        time_span,
        initial_conditions,
        method="BDF",
        rtol=rtol,
        atol=atol_array,
    )

    _, _, _, y1_walls_result, _, _, _ = column.final_wall_values(
        column_grid, bed_properties, left_values, right_values, output_matrix
    )
    T_sim = (
        output_matrix.y[column_grid["num_cells"] : 2 * column_grid["num_cells"]]
        * bed_properties["T_ref"]
    )
    y1_sim = y1_walls_result[-1, :]

    total_error = 0
    error_components = {}

    weights = {"temperature": 1.0, "molefraction": 1.0}

    T_observed = observed_data["temperature"]
    time_temp_observed = observed_data["t_temperature"]
    temp_sim_interp = np.interp(time_temp_observed, output_matrix.t, T_sim[9, :])
    temp_error = np.abs(
        100 / len(T_observed) * np.sum((temp_sim_interp - T_observed) / T_observed)
    )
    error_components["temperature"] = temp_error
    total_error += weights.get("temperature", 1.0) * temp_error

    y1_observed = observed_data["molefraction"]
    t_molefraction_observed = observed_data["t_molefraction"]
    y1_sim_interp = np.interp(t_molefraction_observed, output_matrix.t, y1_sim)
    molefraction_error = np.abs(
        100 / len(y1_observed) * np.sum((y1_sim_interp - y1_observed) / y1_observed)
    )
    error_components["molefraction"] = molefraction_error
    total_error += weights.get("molefraction", 1.0) * molefraction_error

    print(
        f"Params: {dict(zip(param_names, parameters))}, Total Error: {total_error}, Components: {error_components}"
    )

    return total_error


def run_best_simulation(best_params, simulation_conditions, tot_time):
    bed_properties = simulation_conditions["bed_properties"].copy()
    bed_properties.update(best_params)

    column_grid = simulation_conditions["column_grid"]
    initial_conditions = simulation_conditions["initial_conditions"]
    left_values = simulation_conditions["left_values"]
    right_values = simulation_conditions["right_values"]
    column_direction = simulation_conditions["column_direction"]
    rtol = simulation_conditions["rtol"]
    atol_array = simulation_conditions["atol_array"]

    time_span = (0, tot_time)

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

    max_step = 0.1
    first_step = 1e-6
    output_matrix = solve_ivp(
        ODE_func,
        time_span,
        initial_conditions,
        method="BDF",
        rtol=rtol,
        atol=atol_array,
        max_step=max_step,
        first_step=first_step,
    )

    _, _, _, y1_walls_result, _, _, _ = column.final_wall_values(
        column_grid, bed_properties, left_values, right_values, output_matrix
    )

    return output_matrix, y1_walls_result, bed_properties


def plot_results(best_params, simulation_conditions, observed_data, tot_time):
    output_matrix_fitted, y1_walls_result_fitted, bed_properties = run_best_simulation(
        best_params, simulation_conditions, tot_time
    )
    column_grid = simulation_conditions["column_grid"]
    T_sim_fitted = (
        output_matrix_fitted.y[column_grid["num_cells"] : 2 * column_grid["num_cells"]]
        * bed_properties["T_ref"]
    )
    y1_sim_fitted = y1_walls_result_fitted

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(
        output_matrix_fitted.t,
        T_sim_fitted[9, :],
        label="Fitted",
        linewidth=2,
        linestyle="-",
    )
    axes[0].plot(
        observed_data["t_temperature"],
        observed_data["temperature"],
        "ro",
        label="Experimental",
        markersize=4,
    )
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Temperature (K)")
    axes[0].set_title("Temperature Comparison")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        output_matrix_fitted.t,
        y1_sim_fitted[-2, :],
        label="Fitted",
        linewidth=2,
        linestyle="-",
    )
    axes[1].plot(
        observed_data["t_molefraction"],
        observed_data["molefraction"],
        "ro",
        label="Experimental",
        markersize=4,
    )
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Mole Fraction")
    axes[1].set_title("Mole Fraction Comparison")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\nFitted Parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value:.4f}")


if __name__ == "__main__":
    observed_data = import_data()
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

    param_names = ["h_bed", "h_wall", "K_z"]
    initial_guess = [100, 60.0, 0.4]
    bounds = [(50, 150), (25, 75), (0, 1)]
    tot_time = 3000

    print(f"Starting parameter fitting for: {param_names}")
    print(f"Initial guess: {initial_guess}")
    print(f"Bounds: {bounds}")

    result = differential_evolution(
        objective_function,
        bounds,
        seed=42,
        maxiter=1,
        popsize=0,
        atol=1e-6,
        tol=1e-6,
        disp=True,
        init="sobol",
        updating="immediate",
        workers=4,
    )

    print(f"Success: {result.success}")
    print(f"Final parameters: {dict(zip(param_names, result.x))}")
    print(f"Final error: {result.fun}")

    best_params = dict(zip(param_names, result.x))
    best_result = result

    output_matrix_fitted, y1_walls_result_fitted, bed_properties = run_best_simulation(
        best_params, simulation_conditions, tot_time
    )
    column_grid = simulation_conditions["column_grid"]
    T_sim_fitted = (
        output_matrix_fitted.y[column_grid["num_cells"] : 2 * column_grid["num_cells"]]
        * bed_properties["T_ref"]
    )
    y1_sim_fitted = y1_walls_result_fitted

    print("\nFitted Parameters:")
    print(f"Best parameters: {best_params}")
    print(f"Final error: {best_result.fun}")
    print(f"T_sim_fitted: {T_sim_fitted}")
    print(f"y1_sim_fitted: {y1_sim_fitted}")

    # plot_results(best_params, simulation_conditions, observed_data, tot_time)
