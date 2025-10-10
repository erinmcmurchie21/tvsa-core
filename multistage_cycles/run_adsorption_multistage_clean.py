import numpy as np
from scipy.integrate import solve_ivp
from additional_functions_multistage import (
    create_non_uniform_grid, adsorption_isotherm_1, adsorption_isotherm_2, 
    total_mass_balance_error, create_multi_plot, create_quick_plot, 
    product_mass, product_mols, cycle_error, relative_humidity_to_mole_fraction, pressure_ramp, pressure_ramp_2, mole_fraction_to_relative_humidity
)
import time
import tvsa_adsorption_column_multistage as column
import matplotlib.pyplot as plt

from kpi_calculations import (
    calculate_stage_kpis, calculate_cycle_kpis, 
    print_stage_kpis, print_cycle_kpis, export_kpis_to_dict
)

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
        "bed_length": 0.01,                    # Bed length [m]
        "inner_bed_radius": 0.04,           # Inner radius [m]
        "outer_bed_radius": 0.041,            # Outer radius [m]
        "column_area": 0.04**2 * np.pi,     # Cross-sectional area [m²]
        
        # Porosity and density
        "bed_voidage": 0.092,                  # Bed voidage [-]
        "particle_voidage": 0.59,              # Particle voidage [-]
        "total_voidage": 0.092,  # Total voidage [-]
        "bed_density": 55.4,                    # Bed density [kg/m³]
        "sorbent_density": 1590,                # Sorbent density [kg/m³]
        "wall_density": 2700,                  # Wall density [kg/m³]
        
        # Transport properties
        "tortuosity": 3,                       # Tortuosity factor [-]
        "molecular_diffusivity": 0.605e-5,     # Molecular diffusivity [m²/s]
        "particle_diameter": 0.0075,             # Particle diameter [m]
        "K_z": 0.1,                          # Axial dispersion coefficient [m²/s]
        "mu": 1.78e-5,                        # Gas viscosity [Pa·s]
        
        # Mass transfer coefficients
        "mass_transfer_1": 0.0002,             # CO2 mass transfer coeff [s⁻¹]
        "mass_transfer_2": 0.002,              # H2O mass transfer coeff [s⁻¹]
        
        # Heat transfer properties
        "h_bed": 3,                          # Bed heat transfer coeff [W/m²·K]
        "h_wall": 26,                          # Wall heat transfer coeff [W/m²·K]
        "sorbent_heat_capacity": 2070,         # Solid heat capacity [J/kg·K]
        "wall_heat_capacity": 4e6 / 1590,             # Wall heat capacity [J/kg·K]
        "heat_capacity_1": 42.46,          # CO2 adsorbed phase Cp [J/mol·K]
        "heat_capacity_2": 73.1,                 # H2O adsorbed phase Cp [J/mol·K]
        
        # Molecular weights
        "MW_1": 44.01,  # CO2 [g/mol]
        "MW_2": 18.02,  # H2O [g/mol]
        "MW_3": 28.02,  # N2 [g/mol]
        "MW_4": 32.00,  # O2 [g/mol]
        
        # Thermodynamic properties
        "R": 8.314,                           # Universal gas constant [J/mol·K]
        "k": 1.4,                             # Heat capacity ratio [-]
        "ambient_temperature": 293.15,         # Ambient temperature [K]
        "ambient_pressure": 100000,            # Ambient pressure [Pa]

        # Optimisation parameters
        "desorption_temperature": 368.15,      # Desorption temperature [K]
        "vacuum_pressure": 10000,            # Vacuum pressure [Pa]
        "adsorption_time": 13772,              # Adsorption time [s]
        "blowdown_time": 30,                   # Blowdown time [s]
        "heating_time": 704,                  # Heating time [s]
        "desorption_time": 30000,                # Desorption time [s]                  
        "pressurisation_time": 50,               # Pressurisation time [s]

        # Adsorption isotherms
        "isotherm_type_1": "ModifiedToth",  # CO2 isotherm type
        "isotherm_type_2": "GAB",                # H2O isotherm type
        
        # Reference values for scaling (dimensionless variables)
        "P_ref": 101325,    # Reference pressure [Pa]
        "T_ref": 298.15,    # Reference temperature [K]
        "n_ref": 3000,      # Reference adsorbed amount [mol/m³]
        
        # Calculated properties
        "sorbent_mass": 0.01 * 0.04**2 * np.pi * 55.4,          # [kg]
        "sorbent_volume": 0.01 * 0.04**2 * np.pi * (1-0.092),  # [m³]
        "bed_volume": 0.01 * 0.04**2 * np.pi,                   # [m³]
        
        # Efficiencies
        "compressor_efficiency": 0.75,  # Compressor efficiency
        "fan_efficiency": 0.5,        # Fan efficiency

        # Feed conditions
        "feed_velocity": 50 / (0.04**2 * np.pi) / 1e6,      # Superficial feed velocity [m/s]
        "feed_temperature": 293.15,     # Feed temperature [K]


    }
    
    # Create spatial discretization grid
    column_grid = create_non_uniform_grid(bed_properties)
    
    # Initialize state variables
    num_cells = column_grid["num_cells"]
    
    # Initial conditions: ambient pressure, temperature, and composition
    y2_feed = 0.0115
    P_init = np.ones(num_cells) * 100000      # Pressure [Pa]
    T_init = np.ones(num_cells) * 293.15         # Gas temperature [K]
    Tw_init = np.ones(num_cells) * 293.15        # Wall temperature [K]
    y1_init = np.ones(num_cells) * 400e-6  / (1-y2_feed)     # CO2 mole fraction
    y2_init = np.ones(num_cells) * y2_feed       # H2O mole fraction
    y3_init = np.ones(num_cells) * 0.95       # N2 mole fraction
    
    # Calculate initial adsorbed amounts from equilibrium isotherms
    n1_init = adsorption_isotherm_1(P_init, T_init, y1_init, y2_init, y3_init, 400e-6, 0.0115,
                                   bed_properties=bed_properties, 
                                   isotherm_type_1=bed_properties["isotherm_type_1"])[0]
    n2_init = adsorption_isotherm_2(P_init, T_init, y2_init, 
                                   bed_properties=bed_properties, 
                                   isotherm_type=bed_properties["isotherm_type_2"])[0]
    
    # Additional state variables (flow rates and balance errors)
    F_init = np.zeros(8)    # Flow rate variables
    E_init = np.zeros(6)    # Energy balance variables
    
    # Combine all initial conditions (scaled by reference values)
    initial_conditions = np.concatenate([
        P_init / bed_properties["P_ref"],     # Scaled pressure
        T_init / bed_properties["T_ref"],     # Scaled temperature
        Tw_init / bed_properties["T_ref"],    # Scaled wall temperature
        y1_init, y2_init, y3_init,           # Mole fractions (dimensionless)
        n1_init / bed_properties["n_ref"],    # Scaled adsorbed amounts
        n2_init / bed_properties["n_ref"],
        F_init, E_init                        # Additional variables
    ])
    
    # Solver tolerance settings
    rtol = 1e-5  # Relative tolerance
    
    # Absolute tolerances for different variable types
    atol_P = 1e-4 * np.ones(len(P_init))     # Pressure
    atol_T = 1e-4 * np.ones(len(T_init))     # Temperature
    atol_Tw = 1e-4 * np.ones(len(Tw_init))   # Wall temperature
    atol_y1 = 1e-8 * np.ones(len(y1_init))   # CO2 mole fraction
    atol_y2 = 1e-8 * np.ones(len(y2_init))   # H2O mole fraction
    atol_y3 = 1e-8 * np.ones(len(y3_init))   # N2 mole fraction
    atol_n1 = 1e-3 * np.ones(len(n1_init))   # CO2 adsorbed amount
    atol_n2 = 1e-3 * np.ones(len(n2_init))   # H2O adsorbed amount
    atol_F = 1e-4 * np.ones(len(F_init))     # Flow variables
    atol_E = 1e-4 * np.ones(len(E_init))     # Energy variables
    
    atol_array = np.concatenate([atol_P, atol_T, atol_Tw, atol_y1, atol_y2, atol_y3, 
                                atol_n1, atol_n2, atol_F, atol_E])
    
    return bed_properties, column_grid, initial_conditions, rtol, atol_array

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

    # Rest of the function remains the same...
    # Define flow direction and boundary types for each stage
    stage_config = {
        "adsorption": {"left": "mass_flow", "right": "pressure", "direction": "forwards"},
        "blowdown": {"left": "closed", "right": "pressure", "direction": "forwards"},
        "heating": {"left": "closed", "right": "pressure", "direction": "forwards"},
        "desorption": {"left": "closed", "right": "pressure", "direction": "forwards"},
        "pressurisation": {"left": "pressure", "right": "closed", "direction": "forwards"},
        "cooling": {"left": "mass_flow", "right": "closed", "direction": "forwards"}
    }
    
    config = stage_config[stage]
    
    # Standard operating conditions
    feed_velocity = bed_properties["feed_velocity"]
    feed_temperature = bed_properties["feed_temperature"]

    # Feed composition (CO2 capture from air-like mixture)
    if stage == "adsorption" or stage == "pressurisation":
        y2_feed = 0.0115
        feed_composition = {"y1": 0.0004/(1 - y2_feed), "y2": y2_feed, "y3": 0.98}
    else:
        feed_composition = {"y1": None, "y2": None, "y3": None}
    
    # Left boundary conditions
    def get_left_velocity():
        if stage in ["adsorption", "pressurisation"]:
            return feed_velocity
        return None

    def get_left_vol_flow():
        if stage in ["adsorption", "pressurisation"]:
            return feed_velocity * bed_properties["column_area"] * bed_properties["bed_voidage"]
        return None
    
    def get_left_mass_flow():
        if stage in ["adsorption", "pressurisation"]:
            return feed_velocity * bed_properties["column_area"] * bed_properties["bed_voidage"] * 1.13
        return None
    
    left_values = {
        "stage": stage,
        "left_type": config["left"],
        "velocity": get_left_velocity(),
        "left_volume_flow": get_left_vol_flow(),
        "left_mass_flow": get_left_mass_flow(),
        "left_temperature": feed_temperature,
        "y1_left_value": feed_composition["y1"],
        "y2_left_value": feed_composition["y2"],
        "y3_left_value": feed_composition["y3"],
    }

    # Right boundary conditions
    
    right_values = {
        "stage": stage,
        "right_type": config["right"],
        "right_temperature": feed_temperature,
        "y1_right_value": None,
        "y2_right_value": None,
        "y3_right_value": None,
    }
    
    return left_values, right_values, config["direction"], stage

# ============================================================================
# SIMULATION EXECUTION
# ============================================================================

def run_stage(left_values, right_values, column_direction, stage, t_span, 
              initial_conditions, cycle_properties, solver='BDF', ):
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

    #P_initial = initial_conditions[0:column_grid["num_cells"]] * bed_properties["P_ref"]
    print(f"Running {stage} stage...")
    
    bed_properties = cycle_properties['bed_properties']
    column_grid = cycle_properties['column_grid']
    rtol = cycle_properties['rtol']
    atol_array = cycle_properties['atol']

    # Pressure at start of stage to 
    P_initial = initial_conditions[0] * bed_properties["P_ref"] 
    # Define ODE function
    def ODE_func(t, results_vector):
        return column.ODE_calculations(
            t, results_vector=results_vector, column_grid=column_grid, 
            bed_properties=bed_properties, left_values=left_values, 
            right_values=right_values, column_direction=column_direction, stage=stage, P_initial=P_initial
        )
    
    # Solve the ODE system
    # Solve the ODE system
    if stage in ["blowdown", "adsorption", "heating"]:
        max_step = 0.1      # maximum time step in seconds
        first_step = 1e-6   # initial time step in seconds
    else:
        max_step = np.inf
        first_step = None

    t0 = time.time()
    output_matrix = solve_ivp(ODE_func, t_span, initial_conditions, 
                             method=solver, rtol=rtol, atol=atol_array, max_step=max_step,
    first_step=first_step)
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
    T_result = output_matrix.y[num_cells:2*num_cells] * bed_properties["T_ref"]
    Tw_result = output_matrix.y[2*num_cells:3*num_cells] * bed_properties["T_ref"]
    y1_result = output_matrix.y[3*num_cells:4*num_cells]
    y2_result = output_matrix.y[4*num_cells:5*num_cells]
    y3_result = output_matrix.y[5*num_cells:6*num_cells]
    n1_result = output_matrix.y[6*num_cells:7*num_cells] * bed_properties["n_ref"]
    n2_result = output_matrix.y[7*num_cells:8*num_cells] * bed_properties["n_ref"]
    F_result = output_matrix.y[8*num_cells:8*num_cells+8]
    E_result = output_matrix.y[8*num_cells+8:]
    time_array = output_matrix.t
    
    # Calculate mass balance error for validation
    mass_balance_error = total_mass_balance_error(
        F_result, P_result, T_result, n1_result, n2_result, 
        time_array, bed_properties, column_grid
    )
    
    print(f"Completed {stage} stage")
    print(f"Mass balance error: {mass_balance_error}")
    print(f"Simulation time: {simulation_time:.2f} seconds")
    print(f"Stage duration: {time_array[-1]:.2f} seconds")
    print("-------------------------------")
    
    # Calculate wall values for boundary conditions
    (P_walls_result, T_walls_result, Tw_walls_result,
     y1_walls_result, y2_walls_result, y3_walls_result,
     v_walls_result) = column.final_wall_values(column_grid, bed_properties, 
                                               left_values, right_values, output_matrix)
    
    # Extract final conditions for next stage
    P_final = P_result[:, -1]
    T_final = T_result[:, -1]
    Tw_final = Tw_result[:, -1]
    y1_final = y1_result[:, -1]
    y2_final = y2_result[:, -1]
    y3_final = y3_result[:, -1]
    n1_final = n1_result[:, -1]
    n2_final = n2_result[:, -1]
    F_final = np.zeros(8) #F_result[:, -1]
    E_final = np.zeros(6) #E_result[:, -1]
    P_walls_final = P_walls_result[:, -1]
    
    # Prepare final conditions (scaled for next stage)
    final_conditions = np.concatenate([
        P_final / bed_properties["P_ref"], 
        T_final / bed_properties["T_ref"], 
        Tw_final / bed_properties["T_ref"],
        y1_final, y2_final, y3_final, 
        n1_final / bed_properties["n_ref"], 
        n2_final / bed_properties["n_ref"], 
        F_final, 
        E_final
    ])
    
    # Calculate performance metrics
    # Package profile data
    profile_data = {
        'time_array': time_array,
        'P_result': P_result,
        'T_result': T_result,
        'Tw_result': Tw_result,
        'y1_result': y1_result,
        'y2_result': y2_result,
        'y3_result': y3_result,
        'n1_result': n1_result,
        'n2_result': n2_result,
        'P_walls_result': P_walls_result,
    }
    
    # Return simulation results
    return final_conditions, P_walls_final, F_result, E_result, time_array, simulation_time, profile_data


def run_cycle(n_cycles):
    """
    Run complete TVSA cycles until convergence.
    
    Args:
        n_cycles (int): Maximum number of cycles to run
    
    Returns:
        tuple: Simulation profiles and cycle errors
    """
    cycle_properties = {
        'bed_properties': bed_properties,
        'column_grid': column_grid,
        'rtol': rtol,
        'atol': atol_array,
    }
    current_initial_conditions = initial_conditions
    all_cycle_errors = []
    
    # Initialize profile storage
    profiles = {
        'time': [], 'temperature': [], 'pressure_inlet': [], 'pressure_outlet': [],
        'outlet_CO2': [], 'adsorbed_CO2': [], 'outlet_H2O': [], 'adsorbed_H2O': [],
        'wall_temperature': [], 'outlet_N2': [], 'outlet_O2': [],
    }
    # Initialize profile storage
    all_cycle_kpis = []  # Store KPIs for all cycles
    
    
    
    P_walls_final = initial_conditions[0:column_grid["num_cells"]] * bed_properties["P_ref"]
    
    for cycle in range(n_cycles):
        print(f"\n========================================")
        print(f"Starting cycle {cycle + 1} of {n_cycles}")
        
        # Reset profiles for each cycle
        cycle_profiles = {key: [] for key in profiles.keys()}
        time_offset = 0
        
        # Reset E_result at the start of each cycle
        # Dictionary to store stage KPIs
        stage_kpis_dict = {}
        # Define stage sequence and durations
        stages = [
            ("adsorption", [0, bed_properties["adsorption_time"]]),
            ("blowdown", [0, bed_properties["blowdown_time"]]),
            ("heating", [0, bed_properties["heating_time"]]),
            ("desorption", [0, bed_properties["desorption_time"]]),
            ("pressurisation", [0, bed_properties["pressurisation_time"]]),
            #("cooling", [0, 500], 4)
        ]

        stage_conditions = current_initial_conditions
        cycle_time = np.sum([t[1] - t[0] for _, t in stages])
        
        stage_conditions = current_initial_conditions
        
        for stage_name, t_span in stages:
            # Special handling for temperature changes
            if stage_name in ["heating", "desorption"]:
                stage_conditions[2*column_grid["num_cells"]:3*column_grid["num_cells"]] = \
                    bed_properties["desorption_temperature"] / bed_properties["T_ref"]
            elif stage_name == "cooling" or stage_name == "pressurisation" or stage_name == "adsorption" or stage_name == "blowdown":
                stage_conditions[2*column_grid["num_cells"]:3*column_grid["num_cells"]] = \
                    bed_properties["ambient_temperature"] / bed_properties["T_ref"]
              
            E_result = np.zeros((6,))
            F_result = np.zeros((8,))
            simulation_time = np.zeros(6,)
            # Define boundary conditions
            left_vals, right_vals, col_dir, _ = define_boundary_conditions(
                stage_name, bed_properties, P_walls_final[0], P_walls_final[-1]
            )
            
            # Run stage simulation
            stage_results = run_stage(left_vals, right_vals, col_dir, stage_name, 
                                    t_span, stage_conditions, cycle_properties)
            
            if stage_results[0] is None:  # Check for simulation failure
                print(f"Simulation failed at {stage_name} stage")
                return None
            
            # Unpack results - NOW INCLUDES profile_data
            stage_conditions, P_walls_final, F_result, E_result, time_array, simulation_time, profile_data = stage_results

            # CALCULATE STAGE KPIs
            stage_kpis_dict[stage_name] = calculate_stage_kpis(
                stage_name, F_result, time_array, E_result, simulation_time, bed_properties
            )
            # Print stage KPIs
            print_stage_kpis(stage_kpis_dict[stage_name])
            
            # STORE PROFILES (with time offset)
            adjusted_time = time_array + time_offset
            cycle_profiles['time'].extend(adjusted_time)
            cycle_profiles['temperature'].extend(profile_data['T_result'][-1, :])
            cycle_profiles['pressure_inlet'].extend(profile_data['P_walls_result'][0, :])
            cycle_profiles['pressure_outlet'].extend(profile_data['P_walls_result'][-1, :])
            cycle_profiles['outlet_CO2'].extend(profile_data['y1_result'][-1, :])
            cycle_profiles['adsorbed_CO2'].extend(profile_data['n1_result'][-1,:]) #[column_grid["num_cells"]//2, :])
            cycle_profiles['wall_temperature'].extend(profile_data['Tw_result'][-1, :])
            cycle_profiles['adsorbed_H2O'].extend(profile_data['n2_result'][-1,:]) #[column_grid["num_cells"]//2, :])
            cycle_profiles['outlet_H2O'].extend(profile_data['y2_result'][-1, :])
            cycle_profiles['outlet_N2'].extend(profile_data['y3_result'][-1, :])
            cycle_profiles['outlet_O2'].extend(1 - profile_data['y1_result'][-1, :] - profile_data['y2_result'][-1, :] - profile_data['y3_result'][-1, :])
            
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
                print(f"\n{'='*60}")
                print(f"CONVERGED after {cycle + 1} cycles!")
                print(f"{'='*60}\n")
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
    bed_properties, column_grid, initial_conditions, rtol, atol_array = create_fixed_properties()
    
    # Run simulation
    print("Starting TVSA simulation...")
    n_cycles = 3
    
    simulation_results = run_cycle(n_cycles)
    
    if simulation_results is None:
        print("Simulation failed!")
        return


    # Unpack results
    profiles, all_cycle_kpis, all_cycle_errors = simulation_results
    
    # Create visualization
    create_multi_plot(profiles, bed_properties)
    create_quick_plot(np.arange(1, len(all_cycle_errors)+1), 
                     np.log10(all_cycle_errors), "Cycle Convergence", "Log₁₀(Cycle Error)")
    
    print("\nSimulation completed successfully!")

    def save_profile_data(profiles, output_dir="profiles"):
        import os
        os.makedirs(output_dir, exist_ok=True)
        for key, arr in profiles.items():
            filename = f"{output_dir}/{key}.csv"
            np.savetxt(filename, np.array(arr), delimiter=",")

    #save_profile_data(profiles, output_dir="profiles_mechanistic")


if __name__ == "__main__":
    main()