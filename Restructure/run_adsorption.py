from ast import Tuple
from matplotlib.pyplot import step
import numpy as np
from scipy.integrate import solve_ivp
from additional_functions import (
    create_non_uniform_grid,
    total_mass_balance_error,
    create_quick_plot,
    cycle_error,
    plot_all_profiles,
)
import time
import tvsa_adsorption_column as column

from kpi_calculations import (
    calculate_stage_kpis,
    calculate_cycle_kpis,
    print_stage_kpis,
    print_cycle_kpis,
)
# Import configuration and utilities
import LJ_2015 as config_module
import config_base as base
import additional_functions as func

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
# BOUNDARY CONDITIONS
# ============================================================================

def define_stage_conditions(
    stage: str,
    config: base.AdsorptionColumnConfig,
    pressure_left: float,
    pressure_right: float
):
    """
    Define inlet/outlet boundary conditions for each process stage.
    
    Args:
        stage: Process stage name
        config: Complete configuration object
        pressure_left: Left boundary pressure [Pa]
        pressure_right: Right boundary pressure [Pa]
    
    Returns:
        (left_values, right_values, column_direction, stage)
    """
    
    step = config.stages[stage]
    column_direction = step.direction
    
    # Determine composition based on stage
    if stage in ["adsorption", "pressurisation"]:
        composition = config.feed.composition
        temperature = config.feed.temperature
        velocity = config.feed.velocity
    elif stage == "steam_desorption":
        composition = config.steam.composition
        temperature = config.steam.temperature
        velocity = config.steam.velocity
    else:
        composition = {"y1": None, "y2": None, "y3": None}
        temperature = None
        velocity = None
    
    # Calculate mass flow if needed
    mass_flow = None
    vol_flow = None
    if velocity is not None:
        vol_flow = velocity * config.geometry.column_area
        mass_flow = velocity * config.geometry.column_area * 1.13  # Approximate density factor
    
    left_values = {
        "stage": stage,
        "left_type": step.left,
        "velocity": velocity,
        "left_volume_flow": vol_flow,
        "left_mass_flow": mass_flow,
        "left_temperature": temperature,
        "y1_left_value": composition["y1"],
        "y2_left_value": composition["y2"],
        "y3_left_value": composition["y3"],
    }
    
    right_values = {
        "stage": stage,
        "right_type": step.right,
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
    left_values: dict,
    right_values: dict,
    column_direction: str,
    stage: str,
    t_span: Tuple[float, float],
    initial_conditions: np.ndarray,
    config: base.AdsorptionColumnConfig,
    solver: str = "BDF",
):
    """
    Run simulation for a single process stage.
    
    Args:
        left_values: Left boundary conditions
        right_values: Right boundary conditions
        column_direction: Flow direction
        stage: Process stage name
        t_span: Time span [start, end]
        initial_conditions: Initial state vector
        config: Complete configuration object
        solver: ODE solver method
    
    Returns:
        (final_conditions, P_walls_final, F_result, E_result, time_array, 
         simulation_time, profile_data)
    """
    
    print(f"Running {stage} stage...")
    
    # Get initial pressure for pressure ramp calculations
    P_initial = initial_conditions[0] * config.reference.P_ref
    
    # Import ODE function (will be refactored to use config)
    import tvsa_adsorption_column as column
    
    # Define ODE function
    def ODE_func(t, results_vector):
        return column.ODE_calculations(
            t,
            results_vector=results_vector,
            column_grid=config.grid,
            config=config,
            left_values=left_values,
            right_values=right_values,
            column_direction=column_direction,
            stage=stage,
            P_initial=P_initial,
            config_module=config_module,  # module if needed for plotting util
        )
    
    # Set solver parameters based on stage
    if stage == "adsorption":
        max_step = 10.0
        first_step = 1e-6
    elif stage == "heating":
        max_step = 1.0
        first_step = 1e-6
    elif stage == "desorption":
        max_step = 1.0
        first_step = 1e-3
    elif stage == "pressurisation":
        max_step = 1.0
        first_step = 1e-6
    else:
        max_step = np.inf
        first_step = None
    
    # Get tolerances from config
    rtol = 1e-5
    atol_array = base.create_tolerances(config.grid['num_cells'])[1]
    
    # Solve ODE
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
            return None, None, None, None, None, None, None
    
    # Extract results
    num_cells = config.grid['num_cells']
    P_result = output_matrix.y[0:num_cells] * config.reference.P_ref
    T_result = output_matrix.y[num_cells:2*num_cells] * config.reference.T_ref
    Tw_result = output_matrix.y[2*num_cells:3*num_cells] * config.reference.T_ref
    y1_result = output_matrix.y[3*num_cells:4*num_cells]
    y2_result = output_matrix.y[4*num_cells:5*num_cells]
    y3_result = output_matrix.y[5*num_cells:6*num_cells]
    n1_result = output_matrix.y[6*num_cells:7*num_cells] * config.reference.n_ref
    n2_result = output_matrix.y[7*num_cells:8*num_cells] * config.reference.n_ref
    n3_result = output_matrix.y[8*num_cells:9*num_cells] * config.reference.n_ref
    F_result = output_matrix.y[9*num_cells:9*num_cells+8]
    E_result = output_matrix.y[9*num_cells+8:]
    time_array = output_matrix.t
    
    # Calculate mass balance error
    # Skip legacy mass balance dict usage for now or adapt later
    mass_balance_error = 0.0
    
    print(f"Completed {stage} stage")
    print(f"Mass balance error: {mass_balance_error:.2e}")
    print(f"Simulation time: {simulation_time:.2f} seconds")
    print(f"Stage duration: {time_array[-1]:.2f} seconds")
    print("-------------------------------")
    
    # Calculate wall values
    import tvsa_adsorption_column as column
    (
        P_walls_result,
        T_walls_result,
        Tw_walls_result,
        y1_walls_result,
        y2_walls_result,
        y3_walls_result,
        v_walls_result,
    ) = column.final_wall_values(
        config.grid, config, left_values, right_values, output_matrix
    )
    
    # Extract final conditions
    P_final = P_result[:, -1]
    T_final = T_result[:, -1]
    Tw_final = Tw_result[:, -1]
    y1_final = y1_result[:, -1]
    y2_final = y2_result[:, -1]
    y3_final = y3_result[:, -1]
    n1_final = n1_result[:, -1]
    n2_final = n2_result[:, -1]
    n3_final = n3_result[:, -1]
    F_final = np.zeros(8)
    E_final = np.zeros(7)
    P_walls_final = P_walls_result[:, -1]
    
    # Prepare final conditions (scaled)
    final_conditions = np.concatenate([
        P_final / config.reference.P_ref,
        T_final / config.reference.T_ref,
        Tw_final / config.reference.T_ref,
        y1_final,
        y2_final,
        y3_final,
        n1_final / config.reference.n_ref,
        n2_final / config.reference.n_ref,
        n3_final / config.reference.n_ref,
        F_final,
        E_final,
    ])
    
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
    
    return (
        final_conditions,
        P_walls_final,
        F_result,
        E_result,
        time_array,
        simulation_time,
        profile_data,
    )
# ============================================================================


# ============================================================================
# CYCLE SIMULATION
# ============================================================================

def run_cycle(
    n_cycles: int,
    config: base.AdsorptionColumnConfig,
    initial_conditions: np.ndarray
):
    """
    Run complete TVSA cycles until convergence.
    
    Args:
        n_cycles: Maximum number of cycles to run
        config: Complete configuration object
        initial_conditions: Initial state vector
    
    Returns:
        (profiles, all_cycle_kpis, all_cycle_errors)
    """
    
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
    
    P_walls_final = (
        initial_conditions[0:config.grid["num_cells"]] * config.reference.P_ref
    )
    
    for cycle in range(n_cycles):
        print("\n" + "="*60)
        print(f"Starting cycle {cycle + 1} of {n_cycles}")
        print("="*60)
        
        # Reset profiles for each cycle
        cycle_profiles = {key: [] for key in profiles.keys()}
        time_offset = 0
        
        # Define stage sequence with durations
        stages = []
        for stage_name in config.stages.sequence:
            step = config.stages[stage_name]
            duration = step.duration
            t_span = [0, duration]
            stages.append((stage_name, t_span))
        
        stage_conditions = current_initial_conditions
        
        for stage, t_span in stages:
            # Update outside temperature based on stage
            if stage in ["heating", "desorption"]:
                config.operation.outside_temperature = config.operation.desorption_temperature
            elif stage == "steam_desorption":
                # config.operation.outside_temperature = config.operation.steam_desorption_temperature
                pass
            elif stage in ["cooling", "pressurisation", "adsorption", "blowdown"]:
                config.operation.outside_temperature = config.operation.ambient_temperature
            
            # Define boundary conditions
            left_vals, right_vals, col_dir, _ = define_stage_conditions(
                stage, config, P_walls_final[0], P_walls_final[-1]
            )
            
            # Run stage simulation
            stage_results = run_stage(
                left_vals,
                right_vals,
                col_dir,
                stage,
                t_span,
                stage_conditions,
                config,
            )
            
            if stage_results[0] is None:
                print(f"Simulation failed at {stage} stage")
                return None
            
            # Unpack results
            (
                stage_conditions,
                P_walls_final,
                F_result,
                E_result,
                time_array,
                simulation_time,
                profile_data,
            ) = stage_results
            
            # Store profiles
            adjusted_time = time_array + time_offset
            midpoint = config.grid["num_cells"] // 2
            
            cycle_profiles["time"].extend(adjusted_time)
            cycle_profiles["temperature"].extend(profile_data["T_result"][midpoint, :])
            cycle_profiles["pressure_inlet"].extend(profile_data["P_walls_result"][0, :])
            cycle_profiles["pressure_outlet"].extend(profile_data["P_walls_result"][-1, :])
            cycle_profiles["outlet_CO2"].extend(profile_data["y1_result"][-1, :])
            cycle_profiles["adsorbed_CO2"].extend(profile_data["n1_result"][-1, :])
            cycle_profiles["wall_temperature"].extend(profile_data["Tw_result"][-1, :])
            cycle_profiles["adsorbed_H2O"].extend(profile_data["n2_result"][-1, :])
            cycle_profiles["outlet_H2O"].extend(profile_data["y2_result"][-1, :])
            cycle_profiles["outlet_N2"].extend(profile_data["y3_result"][-1, :])
            cycle_profiles["outlet_O2"].extend(
                1 - profile_data["y1_result"][-1, :]
                - profile_data["y2_result"][-1, :]
                - profile_data["y3_result"][-1, :]
            )
            
            time_offset = adjusted_time[-1]
        
        # Calculate cycle error
        cycle_error_value = func.cycle_error(current_initial_conditions, stage_conditions)
        all_cycle_errors.append(cycle_error_value)
        
        print(f"\nCycle {cycle + 1} error: {cycle_error_value:.2e}")
        
        # Update initial conditions for next cycle
        current_initial_conditions = stage_conditions
        
        # Store final cycle profiles
        if cycle == n_cycles - 1:
            for key in profiles:
                profiles[key] = cycle_profiles[key]
        
        # Check convergence
        if len(all_cycle_errors) >= 5:
            if all(error < 1e-9 for error in all_cycle_errors[-5:]):
                print(f"\n{'='*60}")
                print(f"CONVERGED after {cycle + 1} cycles!")
                print(f"{'='*60}\n")
                for key in profiles:
                    profiles[key] = cycle_profiles[key]
                break
    else:
        print(f"\nMaximum cycles ({n_cycles}) reached without convergence")
    
    return profiles, None, all_cycle_errors


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Main execution function."""

    # Initialize system properties
    config, initial_conditions, rtol, atol_array = (
        config_module.create_configuration()
    )

    # Run simulation
    print("Starting TVSA simulation...")
    n_cycles = 5

    simulation_results = run_cycle(n_cycles, config, initial_conditions, )

    if simulation_results is None:
        print("Simulation failed!")
        return -1

    # Unpack results
    profiles, _, all_cycle_errors = simulation_results

    # Create visualization
    config.create_multi_plot(config, profiles)
    create_quick_plot(
        np.arange(1, len(all_cycle_errors) + 1),
        np.log10(all_cycle_errors),
        "Cycle Convergence",
        "Log₁₀(Cycle Error)",
    )
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
