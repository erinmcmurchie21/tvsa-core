import numpy as np
from scipy.integrate import solve_ivp
from additional_functions_multistage import (
    create_non_uniform_grid,
    total_mass_balance_error,
    create_quick_plot,
    cycle_error,
    plot_all_profiles,
)
import time
import tvsa_adsorption_column_multistage as column

from kpi_calculations import (
    calculate_stage_kpis,
    calculate_cycle_kpis,
    print_stage_kpis,
    print_cycle_kpis,
)
import config_LJ_2015 as config

"""
Multi-stage adsorption column simulation for CO2 capture.

This script simulates a Temperature/Vacuum Swing Adsorption (TVSA) process
for CO2 separation from a gas mixture containing CO2, H2O, N2, and O2.

This is a version which uses the same sorbent and cycle properties as that in Stampi-Bombelli, 2020.

The simulation includes five stages:
1. Adsorption - CO2 capture at ambient conditions
2. Blowdown - Pressure reduction
3. Heating - Temperature increase for desorption
4. Pressurisation - Pressure restoration
5. Cooling - Temperature restoration to complete the cycle

Components: (1) CO2, (2) H2O, (3) N2, (4) O2
"""
# ============================================================================
# BOUNDARY CONDITIONS FOR EACH STAGE
# ============================================================================

def define_stage_conditions(stage, bed_properties, pressure_left, pressure_right):
    """
    Define inlet/outlet boundary conditions for each process stage.

    Args:
        stage (str): Process stage name
        bed_properties (dict): Bed properties dictionary
        pressure_left (float): Left boundary pressure [Pa]
        pressure_right (float): Right boundary pressure [Pa]

    Returns:
        tuple: (left_values, right_values, column_direction, stage)
    """

    # Rest of the function remains the same...
    # Define flow direction and boundary types for each stage

    stage_config = bed_properties["stage_config"]
    config = stage_config[stage]
    column_direction = config["direction"]

    # Feed composition (CO2 capture from air-like mixture)
    def get_left_composition():
        if stage == "adsorption" or stage == "pressurisation":
            left_composition = bed_properties["feed_composition"]
        elif stage == "steam_desorption":
            left_composition = bed_properties["steam_composition"]
        else:
            left_composition = {"y1": None, "y2": None, "y3": None}
        return left_composition

    # Left boundary conditions
    def get_left_velocity():
        if stage in ["adsorption", "pressurisation"]:
            left_velocity = bed_properties["feed_velocity"]
        elif stage == "steam_desorption":
            left_velocity = bed_properties["steam_velocity"]
        else:
            left_velocity = None
        return left_velocity

    def get_left_vol_flow():
        if stage in ["adsorption", "pressurisation"]:
            return (
                get_left_velocity()
                * bed_properties["column_area"]
                * bed_properties["bed_voidage"]
            )
        elif stage == "steam_desorption":
            return (
                get_left_velocity()
                * bed_properties["column_area"]
                * bed_properties["bed_voidage"]
            )
        else:
            return None

    def get_left_mass_flow():
        if stage in ["adsorption", "pressurisation"]:
            return (
                get_left_velocity()
                * bed_properties["column_area"]
                * bed_properties["bed_voidage"]
                * 1.13
            )
        elif stage == "steam_desorption":
            return (
                get_left_velocity()
                * bed_properties["column_area"]
                * bed_properties["bed_voidage"]
                * 0.6
            )
        else:
            return None

    def get_left_temperature():
        if stage in ["adsorption", "pressurisation"]:
            left_temperature = bed_properties["feed_temperature"]
        elif stage == "steam_desorption":
            left_temperature = bed_properties["steam_temperature"]
        else:
            left_temperature = None
        return left_temperature

    left_values = {
        "stage": stage,
        "left_type": config["left"],
        "velocity": get_left_velocity(),
        "left_volume_flow": get_left_vol_flow(),
        "left_mass_flow": get_left_mass_flow(),
        "left_temperature": get_left_temperature(),
        "y1_left_value": get_left_composition()["y1"],
        "y2_left_value": get_left_composition()["y2"],
        "y3_left_value": get_left_composition()["y3"],
    }

    # Right boundary conditions

    right_values = {
        "stage": stage,
        "right_type": config["right"],
        "right_temperature": None,
        "y1_right_value": None,
        "y2_right_value": None,
        "y3_right_value": None,
    }

    return left_values, right_values, column_direction, stage


# ============================================================================
# SIMULATION EXECUTION
# ============================================================================


def run_stage(
    left_values,
    right_values,
    column_direction,
    stage,
    t_span,
    initial_conditions,
    cycle_properties,
    solver="LSODA",
):
    """
    Run simulation for a single process stage.

    Args:
        left_values (dict): Left boundary conditions
        right_values (dict): Right boundary conditions
        column_direction (str): Flow direction
        stage (str): Process stage name
        t_span (list): Time span [start, end]
        initial_conditions (array): Initial state vector
        solver (str): ODE solver method

    Returns:
        tuple: Final conditions and simulation results
    """

    # P_initial = initial_conditions[0:column_grid["num_cells"]] * bed_properties["P_ref"]
    print(f"Running {stage} stage...")

    bed_properties = cycle_properties["bed_properties"]
    column_grid = cycle_properties["column_grid"]
    rtol = cycle_properties["rtol"]
    atol_array = cycle_properties["atol"]

    # Pressure at start of stage to
    P_initial = initial_conditions[0] * bed_properties["P_ref"]

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
            stage=stage,
            P_initial=P_initial,
            config=config,
        )

    # Solve the ODE system
    if stage == "adsorption":
        max_step = 10
        first_step = 1e-6
    # elif stage == "blowdown":
    #     max_step = 5.0
    #     first_step = 1e-6
    # elif stage == "heating":
    #     max_step = 1.0  # maximum time step in seconds
    #     first_step = 1e-6  # initial time step in seconds
    # elif stage == "steam_desorption":
    #     max_step = 0.1
    #     first_step = 1e-6
    # elif stage == "desorption":
    #      max_step = 1.0
    #      first_step = 1e-3
    # elif stage == "cooling":
    #     max_step = 0.1
    #     first_step = 1e-6
    elif stage == "pressurisation":
        max_step = 1.0
        first_step = 1e-6
    else:
        max_step = np.inf
        first_step = None

    t0 = time.time()
    output_matrix = solve_ivp(
        ODE_func,
        t_span,
        initial_conditions,
        method=solver,
        rtol=rtol,
        atol=atol_array,
        max_step=max_step,
        first_step=first_step,
    )
    simulation_time = time.time() - t0

    # Check integration status
    if output_matrix.status != 0:
        print(f"WARNING: Integration failed for {stage} stage!")
        print(f"Status: {output_matrix.status}, Message: {output_matrix.message}")
        if output_matrix.status == -1:
            return None, None, None, None, None, None, None, None, None, None, None

    # Extract results and convert back to physical units
    num_cells = column_grid["num_cells"]
    P_result = output_matrix.y[0:num_cells] * bed_properties["P_ref"]
    T_result = output_matrix.y[num_cells : 2 * num_cells] * bed_properties["T_ref"]
    Tw_result = output_matrix.y[2 * num_cells : 3 * num_cells] * bed_properties["T_ref"]
    y1_result = output_matrix.y[3 * num_cells : 4 * num_cells]
    y2_result = output_matrix.y[4 * num_cells : 5 * num_cells]
    y3_result = output_matrix.y[5 * num_cells : 6 * num_cells]
    n1_result = output_matrix.y[6 * num_cells : 7 * num_cells] * bed_properties["n_ref"]
    n2_result = output_matrix.y[7 * num_cells : 8 * num_cells] * bed_properties["n_ref"]
    n3_result = output_matrix.y[8 * num_cells : 9 * num_cells] * bed_properties["n_ref"]
    F_result = output_matrix.y[9 * num_cells : 9 * num_cells + 8]
    E_result = output_matrix.y[9 * num_cells + 8 :]
    time_array = output_matrix.t

    # Calculate mass balance error for validation
    mass_balance_error = total_mass_balance_error(
        F_result,
        P_result,
        T_result,
        n1_result,
        n2_result,
        n3_result,
        time_array,
        bed_properties,
        column_grid,
    )

    
    

    print(f"Completed {stage} stage")
    print(f"Mass balance error: {mass_balance_error}")
    print(f"Simulation time: {simulation_time:.2f} seconds")
    print(f"Stage duration: {time_array[-1]:.2f} seconds")
    print("-------------------------------")

    # Calculate wall values for boundary conditions
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

    # Extract final conditions for next stage
    P_final = P_result[:, -1]
    T_final = T_result[:, -1]
    Tw_final = Tw_result[:, -1]
    y1_final = y1_result[:, -1]
    y2_final = y2_result[:, -1]
    y3_final = y3_result[:, -1]
    n1_final = n1_result[:, -1]
    n2_final = n2_result[:, -1]
    n3_final = n3_result[:, -1]
    F_final = np.zeros(8)  # F_result[:, -1]
    E_final = np.zeros(7)  # E_result[:, -1]
    P_walls_final = P_walls_result[:, -1]

    # Prepare final conditions (scaled for next stage)
    final_conditions = np.concatenate(
        [
            P_final / bed_properties["P_ref"],
            T_final / bed_properties["T_ref"],
            Tw_final / bed_properties["T_ref"],
            y1_final,
            y2_final,
            y3_final,
            n1_final / bed_properties["n_ref"],
            n2_final / bed_properties["n_ref"],
            n3_final / bed_properties["n_ref"],
            F_final,
            E_final,
        ]
    )

    # Calculate performance metrics
    # Package profile data
    profile_data = {
        "time_array": time_array,
        "P_result": P_result,
        "T_result": T_result,
        "Tw_result": Tw_result,
        "y1_result": y1_result,
        "y2_result": y2_result,
        "y3_result": y3_result,
        "n1_result": n1_result,
        "n2_result": n2_result,
        "n3_result": n3_result,
        "P_walls_result": P_walls_result,
    }

    # Return simulation results
    return (
        final_conditions,
        P_walls_final,
        F_result,
        E_result,
        time_array,
        simulation_time,
        profile_data,
    )


def run_cycle(n_cycles):
    """
    Run complete TVSA cycles until convergence.

    Args:
        n_cycles (int): Maximum number of cycles to run

    Returns:
        tuple: Simulation profiles and cycle errors
    """
    cycle_properties = {
        "bed_properties": bed_properties,
        "column_grid": column_grid,
        "rtol": rtol,
        "atol": atol_array,
    }
    current_initial_conditions = initial_conditions
    all_cycle_errors = []

    # Initialize profile storage
    profiles = {
        "time": [],
        "temperature": [],
        "pressure_inlet": [],
        "pressure_outlet": [],
        "outlet_CO2": [],
        "adsorbed_CO2": [],
        "outlet_H2O": [],
        "adsorbed_H2O": [],
        "wall_temperature": [],
        "outlet_N2": [],
        "outlet_O2": [],
    }
    # Initialize profile storage
    all_cycle_kpis = []  # Store KPIs for all cycles

    P_walls_final = (
        initial_conditions[0 : column_grid["num_cells"]] * bed_properties["P_ref"]
    )

    for cycle in range(n_cycles):
        print("\n========================================")
        print(f"Starting cycle {cycle + 1} of {n_cycles}")

        # Reset profiles for each cycle
        cycle_profiles = {key: [] for key in profiles.keys()}
        time_offset = 0

        # Reset E_result at the start of each cycle
        # Dictionary to store stage KPIs
        stage_kpis_dict = {}
        # Define stage sequence and durations

        stage_config = bed_properties["stage_config"]
        stages = [
            (stage, [0, bed_properties[f"{stage}_time"]])
            for stage in stage_config.keys()
            if f"{stage}_time" in bed_properties
        ]

        stage_conditions = current_initial_conditions
        cycle_time = np.sum([t[1] - t[0] for _, t in stages])

        stage_conditions = current_initial_conditions

        for stage_name, t_span in stages:
            # Special handling for temperature changes
            if stage_name in ["heating", "desorption"]:
                bed_properties["outside_temperature"] = bed_properties["desorption_temperature"]
            elif stage_name == "steam_desorption":
                   bed_properties["outside_temperature"] = bed_properties["desorption_temperature"]
            elif stage_name in ["cooling", "pressurisation", "adsorption", "blowdown"]:
                bed_properties["outside_temperature"] = bed_properties["ambient_temperature"]

            E_result = np.zeros((7,))
            F_result = np.zeros((8,))
            simulation_time = np.zeros(
                6,
            )
            # Define boundary conditions
            left_vals, right_vals, col_dir, stage = define_stage_conditions(
                stage_name, bed_properties, P_walls_final[0], P_walls_final[-1]
            )

            # Run stage simulation
            stage_results = run_stage(
                left_vals,
                right_vals,
                col_dir,
                stage_name,
                t_span,
                stage_conditions,
                cycle_properties,
            )

            if stage_results[0] is None:  # Check for simulation failure
                print(f"Simulation failed at {stage_name} stage")
                return None

            # Unpack results - NOW INCLUDES profile_data
            (
                stage_conditions,
                P_walls_final,
                F_result,
                E_result,
                time_array,
                simulation_time,
                profile_data,
            ) = stage_results

            # ...existing code...

            # # Quick plots for pressurisation stage
            # if stage_name == "desorption":
            #     import matplotlib.pyplot as plt

            #     # 1. Outlet Pressure vs Time
            #     plt.figure()
            #     plt.plot(time_array, profile_data["P_result"][-1, :], marker="o")
            #     plt.xlabel("Time (s)")
            #     plt.ylabel("Outlet Pressure (Pa)")
            #     plt.title("Pressurisation: Outlet Pressure vs Time")
            #     plt.grid(True)

            #     # 2. Outlet Temperature vs Time
            #     plt.figure()
            #     plt.plot(time_array, profile_data["T_result"][-1, :], marker="o")
            #     plt.xlabel("Time (s)")
            #     plt.ylabel("Outlet Temperature (K)")
            #     plt.title("Pressurisation: Outlet Temperature vs Time")
            #     plt.grid(True)

            #     # 3. Outlet CO₂ Mole Fraction vs Time
            #     plt.figure()
            #     plt.plot(time_array, profile_data["y1_result"][-1, :], marker="o")
            #     plt.xlabel("Time (s)")
            #     plt.ylabel("Outlet CO₂ Mole Fraction")
            #     plt.title("Pressurisation: Outlet CO₂ Mole Fraction vs Time")
            #     plt.grid(True)

            #     # 4. Outlet Wall Temperature vs Time
            #     plt.figure()
            #     plt.plot(time_array, profile_data["Tw_result"][-1, :], marker="o")
            #     plt.xlabel("Time (s)")
            #     plt.ylabel("Outlet Wall Temperature (K)")
            #     plt.title("Pressurisation: Outlet Wall Temperature vs Time")
            #     plt.grid(True)

            #     plt.show()

# ...existing code...
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.plot(time_array, profile_data["T_result"][-1, :], marker="o")
            # plt.xlabel("Time (s)")
            # plt.ylabel("Temperature (K)")
            # plt.title("Temperature vs Time")
            # plt.grid(True)

            # # 2. Outlet pressure vs time
            # plt.figure()
            # plt.plot(time_array, profile_data["P_result"][-1, :], marker="o")
            # plt.xlabel("Time (s)")
            # plt.ylabel("Outlet Pressure (Pa)")
            # plt.title("Outlet Pressure vs Time")
            # plt.grid(True)

            # # 3. Central node CO2 loading vs time
            # q1 = np.array(profile_data["n1_result"][-1, :]) / bed_properties["bed_density"] * (1 - bed_properties["bed_voidage"])  # mol
            # plt.figure()
            # plt.plot(time_array, q1, marker="o")
            # plt.xlabel("Time (s)")
            # plt.ylabel("CO2 Loading (kg/mol)")
            # plt.title("CO2 Loading vs Time")
            # plt.grid(True)

            # # 4. Outlet CO₂ fraction vs time
            # plt.figure()
            # plt.plot(time_array, profile_data["y1_result"][-1, :], marker="o")
            # plt.xlabel("Time (s)")
            # plt.ylabel("Outlet CO₂ Mole Fraction")
            # plt.title("Outlet CO₂ Fraction vs Time")
            # plt.grid(True)

            # plt.show()

            # CALCULATE STAGE KPIs
            stage_kpis_dict[stage_name] = calculate_stage_kpis(
                stage_name,
                F_result,
                time_array,
                E_result,
                simulation_time,
                bed_properties,
            )
            # Print stage KPIs
            print_stage_kpis(stage_kpis_dict[stage_name])

            # STORE PROFILES (with time offset)
            adjusted_time = time_array + time_offset
            midpoint = column_grid["num_cells"] // 2
            cycle_profiles["time"].extend(adjusted_time)
            cycle_profiles["temperature"].extend(profile_data["T_result"][midpoint, :])
            cycle_profiles["pressure_inlet"].extend(
                profile_data["P_walls_result"][0, :]
            )
            cycle_profiles["pressure_outlet"].extend(
                profile_data["P_walls_result"][-1, :]
            )
            cycle_profiles["outlet_CO2"].extend(profile_data["y1_result"][-1, :])
            cycle_profiles["adsorbed_CO2"].extend(profile_data["n1_result"][-1, :])
            cycle_profiles["wall_temperature"].extend(profile_data["Tw_result"][-1, :])
            cycle_profiles["adsorbed_H2O"].extend(profile_data["n2_result"][-1, :])
            cycle_profiles["outlet_H2O"].extend(profile_data["y2_result"][-1, :])
            cycle_profiles["outlet_N2"].extend(profile_data["y3_result"][-1, :])
            cycle_profiles["outlet_O2"].extend(
                1
                - profile_data["y1_result"][-1, :]
                - profile_data["y2_result"][-1, :]
                - profile_data["y3_result"][-1, :]
            ) 
            
            # Update time offset
            time_offset = adjusted_time[-1]



        # CALCULATE CYCLE KPIs
        cycle_error_value = cycle_error(current_initial_conditions, stage_conditions)
        all_cycle_errors.append(cycle_error_value)

        cycle_kpis = calculate_cycle_kpis(
            cycle + 1, stage_kpis_dict, bed_properties, cycle_error_value
        )
        all_cycle_kpis.append(cycle_kpis)

        # Print comprehensive cycle summary
        print_cycle_kpis(cycle_kpis)

        # Update initial conditions for next cycle
        current_initial_conditions = stage_conditions

        # Store final cycle profiles
        if cycle == n_cycles - 1:  # Last cycle
            for key in profiles:
                profiles[key] = cycle_profiles[key]

        # Check convergence
        if len(all_cycle_errors) >= 5:
            if all(error < 1e-9 for error in all_cycle_errors[-5:]):
                print(f"\n{'=' * 60}")
                print(f"CONVERGED after {cycle + 1} cycles!")
                print(f"{'=' * 60}\n")
                # Store profiles from converged cycle
                for key in profiles:
                    profiles[key] = cycle_profiles[key]
                break
    else:
        print(f"\nMaximum cycles ({n_cycles}) reached without convergence")

    
    return profiles, all_cycle_kpis, all_cycle_errors


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Main execution function."""
    global bed_properties, column_grid, initial_conditions, rtol, atol_array

    # Initialize system properties
    bed_properties, column_grid, initial_conditions, rtol, atol_array = (
        config.create_fixed_properties()
    )

    # Run simulation
    print("Starting TVSA simulation...")
    n_cycles = 5

    simulation_results = run_cycle(n_cycles)

    if simulation_results is None:
        print("Simulation failed!")
        return -1

    # Unpack results
    profiles, all_cycle_kpis, all_cycle_errors = simulation_results

    stage_config = bed_properties["stage_config"]
    stages = [
        (stage, [0, bed_properties[f"{stage}_time"]])
        for stage in stage_config.keys()
        if f"{stage}_time" in bed_properties
    ]
    stage_change_times = np.cumsum([t[1] for _, t in stages])
    stage_names = [name for name, _ in stages]

    # Create visualization
    config.create_multi_plot(profiles, bed_properties)
    create_quick_plot(
        np.arange(1, len(all_cycle_errors) + 1),
        np.log10(all_cycle_errors),
        "Cycle Convergence",
        "Log₁₀(Cycle Error)",
    )
    # plot_all_profiles(
    #     np.array(profiles["time"]),
    #     profiles,
    #     stage_change_times,
    #     stage_names,
    #     bed_properties,
    # )
    print("\nSimulation completed successfully!")

    def save_profile_data(profiles, output_dir="profiles"):
        import os

        os.makedirs(output_dir, exist_ok=True)
        for key, arr in profiles.items():
            filename = f"{output_dir}/{key}.csv"
            np.savetxt(filename, np.array(arr), delimiter=",")

    # save_profile_data(profiles, output_dir="profiles_modifiedtoth_15102025")


if __name__ == "__main__":
    main()
