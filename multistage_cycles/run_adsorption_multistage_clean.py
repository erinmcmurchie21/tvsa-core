import numpy as np
from scipy.integrate import solve_ivp
from additional_functions_multistage import (
    create_non_uniform_grid,
    adsorption_isotherm_1,
    adsorption_isotherm_2,
    total_mass_balance_error,
    create_multi_plot,
    create_quick_plot,
    product_mass,
    cycle_error,
)
import time
import tvsa_adsorption_column_multistage as column

"""
Multi-stage adsorption column simulation for CO2 capture.

This script simulates a Temperature/Vacuum Swing Adsorption (TVSA) process
for CO2 separation from a gas mixture containing CO2, H2O, N2, and O2.

The simulation includes five stages:
1. Adsorption - CO2 capture at ambient conditions
2. Blowdown - Pressure reduction
3. Heating - Temperature increase for desorption
4. Pressurisation - Pressure restoration
5. Cooling - Temperature restoration to complete the cycle

Components: (1) CO2, (2) H2O, (3) N2, (4) O2
"""

# ============================================================================
# CONFIGURATION AND PROPERTIES
# ============================================================================


def create_fixed_properties():
    """
    Define fixed bed properties, grid configuration, and initial conditions.

    Returns:
        tuple: (bed_properties, column_grid, initial_conditions, rtol, atol_array)
    """
    # Column dimensions and physical properties
    bed_properties = {
        # Geometry
        "bed_length": 0.286,  # Bed length [m]
        "inner_bed_radius": 0.00945,  # Inner radius [m]
        "outer_bed_radius": 0.0127,  # Outer radius [m]
        "column_area": 0.00945**2 * np.pi,  # Cross-sectional area [m²]
        # Porosity and density
        "bed_voidage": 0.456,  # Bed voidage [-]
        "particle_voidage": 0.59,  # Particle voidage [-]
        "total_voidage": 0.456 + (1 - 0.456) * 0.59,  # Total voidage [-]
        "bed_density": 435,  # Bed density [kg/m³]
        "sorbent_density": 900,  # Sorbent density [kg/m³]
        "wall_density": 2700,  # Wall density [kg/m³]
        # Transport properties
        "tortuosity": 3,  # Tortuosity factor [-]
        "molecular_diffusivity": 0.605e-5,  # Molecular diffusivity [m²/s]
        "particle_diameter": 2e-3,  # Particle diameter [m]
        "K_z": 0.1,  # Axial dispersion coefficient [m²/s]
        "mu": 1.78e-5,  # Gas viscosity [Pa·s]
        # Mass transfer coefficients
        "mass_transfer_1": 0.0002,  # CO2 mass transfer coeff [s⁻¹]
        "mass_transfer_2": 0.002,  # H2O mass transfer coeff [s⁻¹]
        # Heat transfer properties
        "h_bed": 50,  # Bed heat transfer coeff [W/m²·K]
        "h_wall": 7,  # Wall heat transfer coeff [W/m²·K]
        "sorbent_heat_capacity": 1040,  # Solid heat capacity [J/kg·K]
        "wall_heat_capacity": 902,  # Wall heat capacity [J/kg·K]
        "heat_capacity_1": 840 * 44.01 / 1000,  # CO2 adsorbed phase Cp [J/mol·K]
        "heat_capacity_2": 30,  # H2O adsorbed phase Cp [J/mol·K]
        # Molecular weights
        "MW_1": 44.01,  # CO2 [g/mol]
        "MW_2": 18.02,  # H2O [g/mol]
        "MW_3": 28.02,  # N2 [g/mol]
        "MW_4": 32.00,  # O2 [g/mol]
        # Thermodynamic properties
        "R": 8.314,  # Universal gas constant [J/mol·K]
        "ambient_temperature": 293.15,  # Ambient temperature [K]
        # Adsorption isotherms
        "isotherm_type_1": "Dual_site Langmuir",  # CO2 isotherm type
        "isotherm_type_2": "None",  # H2O isotherm type
        # Reference values for scaling (dimensionless variables)
        "P_ref": 101325,  # Reference pressure [Pa]
        "T_ref": 298.15,  # Reference temperature [K]
        "n_ref": 3000,  # Reference adsorbed amount [mol/m³]
        # Calculated properties
        "sorbent_mass": 0.286 * 0.00945**2 * np.pi * 435,  # [kg]
        "sorbent_volume": 0.286 * 0.00945**2 * np.pi * (1 - 0.456),  # [m³]
        # Efficiencies
        "compressor_efficiency": 0.75,  # Compressor efficiency
    }

    # Create spatial discretization grid
    column_grid = create_non_uniform_grid(bed_properties)

    # Initialize state variables
    num_cells = column_grid["num_cells"]

    # Initial conditions: ambient pressure, temperature, and composition
    P_init = np.ones(num_cells) * 101325  # Pressure [Pa]
    T_init = np.ones(num_cells) * 292  # Gas temperature [K]
    Tw_init = np.ones(num_cells) * 292  # Wall temperature [K]
    y1_init = np.ones(num_cells) * 0.02  # CO2 mole fraction
    y2_init = np.ones(num_cells) * 1e-6  # H2O mole fraction
    y3_init = np.ones(num_cells) * 0.95  # N2 mole fraction

    # Calculate initial adsorbed amounts from equilibrium isotherms
    n1_init = adsorption_isotherm_1(
        P_init,
        T_init,
        y1_init,
        y2_init,
        y3_init,
        1e-6,
        bed_properties=bed_properties,
        isotherm_type_1=bed_properties["isotherm_type_1"],
    )[0]
    n2_init = adsorption_isotherm_2(
        P_init,
        T_init,
        y2_init,
        bed_properties=bed_properties,
        isotherm_type=bed_properties["isotherm_type_2"],
    )[0]

    # Additional state variables (flow rates and balance errors)
    F_init = np.zeros(8)  # Flow rate variables
    E_init = np.zeros(5)  # Energy balance variables

    # Combine all initial conditions (scaled by reference values)
    initial_conditions = np.concatenate(
        [
            P_init / bed_properties["P_ref"],  # Scaled pressure
            T_init / bed_properties["T_ref"],  # Scaled temperature
            Tw_init / bed_properties["T_ref"],  # Scaled wall temperature
            y1_init,
            y2_init,
            y3_init,  # Mole fractions (dimensionless)
            n1_init / bed_properties["n_ref"],  # Scaled adsorbed amounts
            n2_init / bed_properties["n_ref"],
            F_init,
            E_init,  # Additional variables
        ]
    )

    # Solver tolerance settings
    rtol = 1e-4  # Relative tolerance

    # Absolute tolerances for different variable types
    atol_P = 1e-4 * np.ones(len(P_init))  # Pressure
    atol_T = 1e-4 * np.ones(len(T_init))  # Temperature
    atol_Tw = 1e-4 * np.ones(len(Tw_init))  # Wall temperature
    atol_y1 = 1e-8 * np.ones(len(y1_init))  # CO2 mole fraction
    atol_y2 = 1e-8 * np.ones(len(y2_init))  # H2O mole fraction
    atol_y3 = 1e-8 * np.ones(len(y3_init))  # N2 mole fraction
    atol_n1 = 1e-3 * np.ones(len(n1_init))  # CO2 adsorbed amount
    atol_n2 = 1e-3 * np.ones(len(n2_init))  # H2O adsorbed amount
    atol_F = 1e-4 * np.ones(len(F_init))  # Flow variables
    atol_E = 1e-4 * np.ones(len(E_init))  # Energy variables

    atol_array = np.concatenate(
        [
            atol_P,
            atol_T,
            atol_Tw,
            atol_y1,
            atol_y2,
            atol_y3,
            atol_n1,
            atol_n2,
            atol_F,
            atol_E,
        ]
    )

    return bed_properties, column_grid, initial_conditions, rtol, atol_array


# ============================================================================
# PRESSURE PROFILES FOR DIFFERENT STAGES
# ============================================================================


def pressure_ramp(t, stage, pressure_previous_stage):
    """
    Define pressure profiles for different process stages.

    Args:
        t (float): Current time [s]
        stage (str): Process stage name
        pressure_previous_stage (float): Pressure from previous stage [Pa]

    Returns:
        float: Pressure at time t [Pa]
    """
    if stage == "adsorption":
        # Maintain atmospheric pressure during adsorption
        return 101325

    elif stage == "blowdown":
        # Exponential pressure drop from atmospheric to vacuum
        initial_pressure = pressure_previous_stage
        final_pressure = 40000  # Target vacuum pressure [Pa]
        tau = 1 - 1 / np.e  # Time constant
        return final_pressure + (initial_pressure - final_pressure) * np.exp(-t / tau)

    elif stage == "heating":
        # Maintain vacuum pressure during heating
        initial_pressure = pressure_previous_stage
        final_pressure = 40000
        tau = 1 - 1 / np.e
        return final_pressure + (initial_pressure - final_pressure) * np.exp(-t / tau)

    elif stage == "pressurisation":
        # Rapid pressurization back to atmospheric
        initial_pressure = pressure_previous_stage
        final_pressure = 101325  # Slightly above atmospheric [Pa]
        tau = 0.2  # Fast time constant
        return final_pressure - (final_pressure - initial_pressure) * np.exp(-t / tau)

    elif stage == "cooling":
        # Return to atmospheric pressure
        initial_pressure = pressure_previous_stage
        final_pressure = 101325  # Slightly above atmospheric [Pa]
        tau = 0.2  # Fast time constant
        return (
            final_pressure  # - (final_pressure - initial_pressure) * np.exp(-t / tau)
        )


# ============================================================================
# BOUNDARY CONDITIONS FOR EACH STAGE
# ============================================================================


def define_boundary_conditions(stage, bed_properties, pressure_left, pressure_right):
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
    # Define flow direction and boundary types for each stage
    stage_config = {
        "adsorption": {
            "left": "mass_flow",
            "right": "pressure",
            "direction": "forwards",
        },
        "blowdown": {"left": "closed", "right": "pressure", "direction": "forwards"},
        "heating": {"left": "closed", "right": "pressure", "direction": "forwards"},
        "pressurisation": {
            "left": "pressure",
            "right": "closed",
            "direction": "forwards",
        },
        "cooling": {"left": "mass_flow", "right": "pressure", "direction": "forwards"},
    }

    config = stage_config[stage]

    # Standard operating conditions
    operating_velocity = (
        100 / 60 / 1e6 / bed_properties["column_area"] / bed_properties["bed_voidage"]
    )  # [m/s]
    operating_temperature = 293.15  # [K]

    # Feed composition (CO2 capture from air-like mixture)
    feed_composition = {"y1": 0.15, "y2": 1e-6, "y3": 0.84}  # CO2, H2O, N2

    # Product composition (high purity CO2)
    product_composition = {"y1": 0.9999, "y2": 1e-6, "y3": 1e-6}

    # Left boundary conditions
    def get_left_velocity():
        if stage in ["adsorption", "cooling"]:
            return operating_velocity
        return 0  # No flow for closed boundary

    def get_left_pressure_func():
        if stage == "pressurisation":
            return lambda t: pressure_ramp(t, "pressurisation", pressure_left)
        return lambda t: None  # Not used for mass flow boundaries

    left_values = {
        "left_type": config["left"],
        "velocity": get_left_velocity(),
        "left_volume_flow": 1.6667e-6,  # [m³/s]
        "left_mass_flow": 0.01 * bed_properties["column_area"] * 1.13,  # [kg/s]
        "left_temperature": operating_temperature,
        "left_pressure": get_left_pressure_func(),
        "y1_left_value": feed_composition["y1"],
        "y2_left_value": feed_composition["y2"],
        "y3_left_value": feed_composition["y3"],
    }

    # Right boundary conditions
    def get_right_pressure_func():
        if config["right"] == "pressure":
            return lambda t: pressure_ramp(t, stage, pressure_right)
        return lambda t: None  # Not used for closed boundaries

    right_values = {
        "right_type": config["right"],
        "right_pressure_func": get_right_pressure_func(),
        "right_temperature": operating_temperature,
        "y1_right_value": product_composition["y1"],
        "y2_right_value": product_composition["y2"],
        "y3_right_value": product_composition["y3"],
    }

    return left_values, right_values, config["direction"], stage


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
    solver="BDF",
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
        )

    # Solve the ODE system
    t0 = time.time()
    output_matrix = solve_ivp(
        ODE_func, t_span, initial_conditions, method=solver, rtol=rtol, atol=atol_array
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
    F_result = output_matrix.y[8 * num_cells : 8 * num_cells + 8]
    E_result = output_matrix.y[8 * num_cells + 8 :]
    time_array = output_matrix.t

    # Calculate mass balance error for validation
    mass_balance_error = total_mass_balance_error(
        F_result,
        P_result,
        T_result,
        n1_result,
        n2_result,
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
    F_final = np.zeros(8)  # F_result[:, -1]
    E_final = np.zeros(5)  # E_result[:, -1]
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
            F_final,
            E_final,
        ]
    )

    # Calculate performance metrics
    mols_CO2_out_st, mols_carrier_gas_out_st, mols_CO2_in_st = product_mass(
        F_result, time_array, bed_properties
    )

    stage_energy = E_result[:, -1]

    # Return simulation results
    return (
        final_conditions,
        time_array,
        P_result,
        T_result,
        Tw_result,
        y1_result,
        n1_result,
        E_final,
        P_walls_result,
        mols_CO2_out_st,
        mols_carrier_gas_out_st,
        mols_CO2_in_st,
        stage_energy,
    )


def run_cycle(n_cycles):
    """
    Run complete TVSA cycles until convergence.

    Args:
        n_cycles (int): Maximum number of cycles to run

    Returns:
        tuple: Simulation profiles and cycle errors
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
        "wall_temperature": [],
        "mols_CO2_out": [],
        "mols_carrier_gas_out": [],
        "mols_CO2_in": [],
        "thermal_energy_input": [],
        "vacuum_energy_input": [],
    }

    P_walls_final = (
        initial_conditions[0 : column_grid["num_cells"]] * bed_properties["P_ref"]
    )

    for cycle in range(n_cycles):
        print("\n========================================")
        print(f"Starting cycle {cycle + 1} of {n_cycles}")

        # Reset profiles for each cycle
        cycle_profiles = {key: [] for key in profiles.keys()}
        time_offset = 0

        # Define stage sequence and durations
        stages = [
            ("adsorption", [0, 200]),
            ("blowdown", [0, 50]),
            ("heating", [0, 1000]),
            ("pressurisation", [0, 30]),
            ("cooling", [0, 500]),
        ]

        stage_conditions = current_initial_conditions

        for stage_name, t_span in stages:
            # Special handling for heating stage (increase wall temperature)
            if stage_name == "heating":
                stage_conditions[
                    2 * column_grid["num_cells"] : 3 * column_grid["num_cells"]
                ] = 400 / bed_properties["T_ref"]

            # Define boundary conditions
            left_vals, right_vals, col_dir, _ = define_boundary_conditions(
                stage_name, bed_properties, P_walls_final[0], P_walls_final[-1]
            )

            # Run stage simulation
            stage_results = run_stage(
                left_vals, right_vals, col_dir, stage_name, t_span, stage_conditions
            )

            if stage_results[0] is None:  # Check for simulation failure
                print(f"Simulation failed at {stage_name} stage")
                return None

            # Unpack results
            (
                stage_conditions,
                time_array,
                P_result,
                T_result,
                Tw_result,
                y1_result,
                n1_result,
                E_final,
                P_walls_result,
                mols_CO2_out_st,
                mols_carrier_gas_out_st,
                mols_CO2_in_st,
                stage_energy,
            ) = stage_results

            # Update wall pressures for next stage
            P_walls_final = P_walls_result[:, -1]

            # Store profiles with time offset
            adjusted_time = time_array + time_offset
            cycle_profiles["time"].extend(adjusted_time)
            cycle_profiles["temperature"].extend(
                T_result[column_grid["num_cells"] // 2, :]
            )
            cycle_profiles["pressure_inlet"].extend(P_walls_result[0, :])
            cycle_profiles["pressure_outlet"].extend(P_walls_result[-1, :])
            cycle_profiles["outlet_CO2"].extend(y1_result[-1, :])  # Assuming outlet CO2
            cycle_profiles["adsorbed_CO2"].extend(
                n1_result[column_grid["num_cells"] // 2, :]
            )
            cycle_profiles["wall_temperature"].extend(
                Tw_result[column_grid["num_cells"] // 2, :]
            )

            # Update time offset
            time_offset = adjusted_time[-1]

            # Store mass flows (per stage)
            cycle_profiles["mols_CO2_out"].append(mols_CO2_out_st)
            cycle_profiles["mols_carrier_gas_out"].append(mols_carrier_gas_out_st)
            cycle_profiles["mols_CO2_in"].append(mols_CO2_in_st)
            cycle_profiles["thermal_energy_input"].append(stage_energy[3])
            cycle_profiles["vacuum_energy_input"].append(stage_energy[4])

        # Calculate cycle convergence
        cycle_error_value = cycle_error(current_initial_conditions, stage_conditions)
        all_cycle_errors.append(cycle_error_value)

        # Update initial conditions for next cycle
        current_initial_conditions = stage_conditions

        # Calculate and display performance metrics
        heating_stage_idx = 2  # Index of heating stage
        cycle_purity = cycle_profiles["mols_CO2_out"][heating_stage_idx] / (
            cycle_profiles["mols_CO2_out"][heating_stage_idx]
            + cycle_profiles["mols_carrier_gas_out"][heating_stage_idx]
        )

        recovery_rate = cycle_profiles["mols_CO2_out"][heating_stage_idx] / sum(
            cycle_profiles["mols_CO2_in"]
        )

        thermal_energy_input = cycle_profiles["thermal_energy_input"][
            2
        ]  # Heating stage
        vacuum_energy_input = np.sum(
            cycle_profiles["vacuum_energy_input"]
        )  # Blowdown and Heating stages

        print(f"Cycle {cycle + 1} Results:")
        print(f"  Purity: {cycle_purity:.6f}")
        print(f"  Recovery Rate: {recovery_rate:.6f}")
        print(f"  Cycle Error: {cycle_error_value}")
        print("  Thermal Energy Input (J): ", thermal_energy_input)
        print("  Vacuum Energy Input (J): ", vacuum_energy_input)

        # Store final cycle profiles
        if cycle == n_cycles - 1:  # Last cycle
            for key in profiles:
                profiles[key] = cycle_profiles[key]

        # Check convergence
        if len(all_cycle_errors) >= 5:
            if all(error < 1e-9 for error in all_cycle_errors[-5:]):
                print(f"\nConverged after {cycle + 1} cycles!")
                break
    else:
        print(f"\nMaximum cycles ({n_cycles}) reached without convergence")

    return (
        profiles["temperature"],
        profiles["outlet_CO2"],
        profiles["time"],
        profiles["pressure_inlet"],
        profiles["pressure_outlet"],
        profiles["adsorbed_CO2"],
        profiles["wall_temperature"],
        all_cycle_errors,
    )


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Main execution function."""
    global bed_properties, column_grid, initial_conditions, rtol, atol_array

    # Initialize system properties
    bed_properties, column_grid, initial_conditions, rtol, atol_array = (
        create_fixed_properties()
    )

    # Run simulation
    print("Starting TVSA simulation...")
    n_cycles = 10

    simulation_results = run_cycle(n_cycles)

    if simulation_results is None:
        print("Simulation failed!")
        return

    # Unpack results
    (
        temperature_profile,
        outlet_CO2_profile,
        time_profile,
        pressure_profile_inlet,
        pressure_profile_outlet,
        adsorbed_CO2_profile,
        wall_temperature_profile,
        all_cycle_errors,
    ) = simulation_results

    # Generate plots
    plots = [
        (
            time_profile,
            temperature_profile,
            "Temperature at Column Midpoint",
            "Temperature (K)",
        ),
        (time_profile, pressure_profile_inlet, "Inlet Pressure", "Pressure (Pa)"),
        (time_profile, pressure_profile_outlet, "Outlet Pressure", "Pressure (Pa)"),
        (time_profile, outlet_CO2_profile, "Outlet CO2 Concentration", "Mole Fraction"),
        (
            time_profile,
            adsorbed_CO2_profile,
            "Adsorbed CO2 at Midpoint",
            "Concentration (mol/m³)",
        ),
        (
            time_profile,
            wall_temperature_profile,
            "Wall Temperature at Midpoint",
            "Temperature (K)",
        ),
    ]

    # Create visualization
    create_multi_plot(plots, ncols=3)
    create_quick_plot(
        np.arange(1, len(all_cycle_errors) + 1),
        np.log10(all_cycle_errors),
        "Cycle Convergence",
        "Log₁₀(Cycle Error)",
    )

    print("\nSimulation completed successfully!")


if __name__ == "__main__":
    main()
