import numpy as np
from scipy.integrate import solve_ivp
from additional_functions_multistage import (
    create_non_uniform_grid, adsorption_isotherm_1, adsorption_isotherm_2, 
    total_mass_balance_error, create_multi_plot, create_quick_plot, 
    product_mass, cycle_error, relative_humidity_to_mole_fraction
)
import time
import tvsa_adsorption_column_multistage as column
import matplotlib.pyplot as plt

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
        "ambient_pressure": 101325,            # Ambient pressure [Pa]

        # Optimisation parameters
        "desorption_temperature": 368.15,      # Desorption temperature [K]
        "vacuum_pressure": 50000,            # Vacuum pressure [Pa]
        "adsorption_time": 13772,              # Adsorption time [s]
        "blowdown_time": 30,                   # Blowdown time [s]
        "heating_time": 704,                  # Heating time [s]
        "desorption_time": 2000,                # Desorption time [s]
        "pressurisation_time": 50,               # Pressurisation time [s]

        # Adsorption isotherms
        "isotherm_type_1": "Toth",  # CO2 isotherm type
        "isotherm_type_2": "GAB",                # H2O isotherm type
        
        # Reference values for scaling (dimensionless variables)
        "P_ref": 101325,    # Reference pressure [Pa]
        "T_ref": 298.15,    # Reference temperature [K]
        "n_ref": 3000,      # Reference adsorbed amount [mol/m³]
        
        # Calculated properties
        "sorbent_mass": 0.286 * 0.00945**2 * np.pi * 55.4,          # [kg]
        "sorbent_volume": 0.286 * 0.00945**2 * np.pi * (1-0.092),  # [m³]
        
        # Efficiencies
        "compressor_efficiency": 0.75,  # Compressor efficiency

    }
    
    # Create spatial discretization grid
    column_grid = create_non_uniform_grid(bed_properties)
    
    # Initialize state variables
    num_cells = column_grid["num_cells"]
    
    # Initial conditions: ambient pressure, temperature, and composition
    P_init = np.ones(num_cells) * 101325      # Pressure [Pa]
    T_init = np.ones(num_cells) * 292         # Gas temperature [K]
    Tw_init = np.ones(num_cells) * 292        # Wall temperature [K]
    y1_init = np.ones(num_cells) * 1e-6     # CO2 mole fraction
    y2_init = np.ones(num_cells) * relative_humidity_to_mole_fraction(0.4, P_init, T_init)       # H2O mole fraction
    y3_init = np.ones(num_cells) * 0.95       # N2 mole fraction
    
    # Calculate initial adsorbed amounts from equilibrium isotherms
    n1_init = adsorption_isotherm_1(P_init, T_init, y1_init, y2_init, y3_init, 1e-6, 1e-6,
                                   bed_properties=bed_properties, 
                                   isotherm_type_1=bed_properties["isotherm_type_1"])[0]
    n2_init = adsorption_isotherm_2(P_init, T_init, y2_init, 
                                   bed_properties=bed_properties, 
                                   isotherm_type=bed_properties["isotherm_type_2"])[0]
    
    # Additional state variables (flow rates and balance errors)
    F_init = np.zeros(8)    # Flow rate variables
    E_init = np.zeros(5)    # Energy balance variables
    
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
        return bed_properties["ambient_pressure"]
        
    elif stage == "blowdown":
        # Exponential pressure drop from atmospheric to vacuum
        initial_pressure = pressure_previous_stage
        final_pressure = bed_properties["vacuum_pressure"]  # Target vacuum pressure [Pa]
        tau = 1 - 1/np.e       # Time constant
        return final_pressure + (initial_pressure - final_pressure) * np.exp(-t / tau)
        
    elif stage == "heating":
        # Maintain vacuum pressure during heating
        initial_pressure = pressure_previous_stage
        final_pressure = bed_properties["vacuum_pressure"]
        tau = 2
        return final_pressure + (initial_pressure - final_pressure) * np.exp(-t / tau)
    
    elif stage == "desorption":
        # Maintain vacuum pressure during desorption
        initial_pressure = pressure_previous_stage
        final_pressure = bed_properties["vacuum_pressure"]
        tau = 2
        return final_pressure
        
    elif stage == "pressurisation":
        # Rapid pressurization back to atmospheric
        initial_pressure = pressure_previous_stage
        final_pressure = bed_properties["ambient_pressure"]  # Slightly above atmospheric [Pa]
        tau = 0.2               # Fast time constant
        return final_pressure - (final_pressure - initial_pressure) * np.exp(-t / tau)
        
    elif stage == "cooling":
        # Return to atmospheric pressure
        initial_pressure = pressure_previous_stage
        final_pressure = 101325  # Slightly above atmospheric [Pa]
        tau = 0.2               # Fast time constant
        return final_pressure  # - (final_pressure - initial_pressure) * np.exp(-t / tau)


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
        "adsorption": {"left": "mass_flow", "right": "pressure", "direction": "forwards"},
        "blowdown": {"left": "closed", "right": "pressure", "direction": "forwards"},
        "heating": {"left": "closed", "right": "pressure", "direction": "forwards"},
        "desorption": {"left": "closed", "right": "pressure", "direction": "forwards"},
        "pressurisation": {"left": "pressure", "right": "closed", "direction": "forwards"},
        "cooling": {"left": "mass_flow", "right": "pressure", "direction": "forwards"}
    }
    
    config = stage_config[stage]
    
    # Standard operating conditions
    feed_velocity = 50 / 1e6 / bed_properties["column_area"] / bed_properties["bed_voidage"]  # [m/s]
    feed_temperature = 293.15  # [K]
    
    
    # Feed composition (CO2 capture from air-like mixture)
    feed_composition = {"y1": 0.0004, "y2": 0.0115, "y3": 0.98}  # CO2, H2O, N2
    
    # Product composition (high purity CO2)
    product_composition = {"y1": 0.9999, "y2": 1e-6, "y3": 1e-6}
    
    # Left boundary conditions
    def get_left_velocity():
        if stage in ["adsorption", "cooling"]:
            return feed_velocity
        return 0  # No flow for closed boundary

    def get_left_vol_flow():
        if get_left_velocity() > 0:
            return feed_velocity * bed_properties["column_area"] * bed_properties["bed_voidage"]
        return 0

    def get_left_pressure_func():
        if stage == "pressurisation":
            return lambda t: pressure_ramp(t, "pressurisation", pressure_left)
        return lambda t: None  # Not used for mass flow boundaries
    
    left_values = {
        "left_type": config["left"],
        "velocity": get_left_velocity(),
        "left_volume_flow": get_left_vol_flow(),  # [m³/s]
        "left_mass_flow": 0.01 * bed_properties["column_area"] * 1.13,  # [kg/s]
        "left_temperature": feed_temperature,
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
        "right_temperature": feed_temperature,
        "y1_right_value": product_composition["y1"],
        "y2_right_value": product_composition["y2"],
        "y3_right_value": product_composition["y3"],
    }
    
    return left_values, right_values, config["direction"], stage


# ============================================================================
# SIMULATION EXECUTION
# ============================================================================

def run_stage(left_values, right_values, column_direction, stage, t_span, 
              initial_conditions, solver='BDF'):
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
    t0 = time.time()
    output_matrix = solve_ivp(ODE_func, t_span, initial_conditions, 
                             method=solver, rtol=rtol, atol=atol_array)
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
    E_final = np.zeros(5) #E_result[:, -1]
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
    mass_CO2_out_st, mass_carrier_gas_out_st, mass_CO2_in_st, mass_H2O_out_st = product_mass(
        F_result, time_array, bed_properties
    )

    stage_energy = E_result[:, -1]
    fan_energy = (1 / bed_properties["compressor_efficiency"] * (bed_properties["k"]/(bed_properties["k"]-1)) * bed_properties["ambient_temperature"] 
                  * ((P_walls_final[0]/bed_properties["ambient_pressure"])**((bed_properties["k"] - 1) - 1) / (1 - bed_properties["k"]) - 1)
                  * (F_result[0]+F_result[1]+F_result[2]+F_result[3]))  # Fan energy for pressurisation
    stage_energy = np.append(stage_energy, fan_energy)  # Append fan energy to stage energy
    
    # Return simulation results
    return (final_conditions, time_array, P_result, T_result, Tw_result,
            y1_result, y2_result, n1_result, n2_result, E_final, P_walls_result, mass_CO2_out_st, 
            mass_carrier_gas_out_st, mass_CO2_in_st, mass_H2O_out_st, stage_energy)


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
        'time': [], 'temperature': [], 'pressure_inlet': [], 'pressure_outlet': [],
        'outlet_CO2': [], 'adsorbed_CO2': [], 'outlet_H2O': [], 'adsorbed_H2O': [], 
        'wall_temperature': [], 'mass_CO2_out': [], 'mass_carrier_gas_out': [], 
        'mass_CO2_in': [], 'mass_H2O_out': [], 'thermal_energy_input': [], 'vacuum_energy_input': [], 
        'fan_energy_input': []
    }
    
    P_walls_final = initial_conditions[0:column_grid["num_cells"]] * bed_properties["P_ref"]
    
    for cycle in range(n_cycles):
        print(f"\n========================================")
        print(f"Starting cycle {cycle + 1} of {n_cycles}")
        
        # Reset profiles for each cycle
        cycle_profiles = {key: [] for key in profiles.keys()}
        time_offset = 0
        
        # Define stage sequence and durations
        stages = [
            ("adsorption", [0, bed_properties["adsorption_time"]]),
            ("blowdown", [0, bed_properties["blowdown_time"]]),
            ("heating", [0, bed_properties["heating_time"]]),
            ("desorption", [0, bed_properties["desorption_time"]]),
            ("pressurisation", [0, bed_properties["pressurisation_time"]]),
            #("cooling", [0, 500], 4)
        ]

        cycle_time = np.sum([t[1] - t[0] for _, t in stages])

        stage_key = {
            "adsorption": 0,
            "blowdown": 1,
            "heating": 2,
            "desorption": 3,
            "pressurisation": 4,
            "cooling": 5
        }
        
        stage_conditions = current_initial_conditions
        
        for stage_name, t_span in stages:
            # Special handling for heating stage (increase wall temperature)
            if stage_name == "heating" or stage_name == "desorption":
                stage_conditions[2*column_grid["num_cells"]:3*column_grid["num_cells"]] = bed_properties["desorption_temperature"]/bed_properties["T_ref"]
                

            # Define boundary conditions
            left_vals, right_vals, col_dir, _ = define_boundary_conditions(
                stage_name, bed_properties, P_walls_final[0], P_walls_final[-1]
            )
            
            # Run stage simulation
            stage_results = run_stage(left_vals, right_vals, col_dir, stage_name, 
                                    t_span, stage_conditions)
            
            if stage_results[0] is None:  # Check for simulation failure
                print(f"Simulation failed at {stage_name} stage")
                return None
            
            # Unpack results
            (stage_conditions, time_array, P_result, T_result, Tw_result,
             y1_result, y2_result, n1_result, n2_result, E_final, P_walls_result, mass_CO2_out_st, 
             mass_carrier_gas_out_st, mass_CO2_in_st, mass_H2O_out_st, stage_energy) = stage_results

            # Update wall pressures for next stage
            P_walls_final = P_walls_result[:, -1]
            
            # Store profiles with time offset
            adjusted_time = time_array + time_offset
            cycle_profiles['time'].extend(adjusted_time)
            cycle_profiles['temperature'].extend(T_result[column_grid["num_cells"]//2, :])
            cycle_profiles['pressure_inlet'].extend(P_walls_result[0, :])
            cycle_profiles['pressure_outlet'].extend(P_walls_result[-1, :])
            cycle_profiles['outlet_CO2'].extend(y1_result[-1, :])  # Assuming outlet CO2
            cycle_profiles['adsorbed_CO2'].extend(n1_result[column_grid["num_cells"]//2, :])
            cycle_profiles['wall_temperature'].extend(Tw_result[column_grid["num_cells"]//2, :])
            cycle_profiles['adsorbed_H2O'].extend(n2_result[column_grid["num_cells"]//2, :])
            cycle_profiles['outlet_H2O'].extend(y2_result[-1, :])  # Assuming outlet H2O

            # Update time offset
            time_offset = adjusted_time[-1]
            
            # Store mass flows (per stage)
            cycle_profiles['mass_CO2_out'].append(mass_CO2_out_st)
            cycle_profiles['mass_carrier_gas_out'].append(mass_carrier_gas_out_st)
            cycle_profiles['mass_CO2_in'].append(mass_CO2_in_st)
            cycle_profiles['mass_H2O_out'].append(mass_H2O_out_st)
            cycle_profiles['thermal_energy_input'].append(stage_energy[3])
            cycle_profiles['vacuum_energy_input'].append(stage_energy[4])
            cycle_profiles['fan_energy_input'].append(stage_energy[5])

        # Calculate cycle convergence
        cycle_error_value = cycle_error(current_initial_conditions, stage_conditions)
        all_cycle_errors.append(cycle_error_value)
        
        # Update initial conditions for next cycle
        current_initial_conditions = stage_conditions
        
        # Calculate and display performance metrics
        heating_stage_idx = 2  # Index of heating stage
        desorption_stage_idx = 3  # Index of desorption stage
        cycle_purity = (cycle_profiles['mass_CO2_out'][desorption_stage_idx] /
                       (cycle_profiles['mass_CO2_out'][desorption_stage_idx] +
                        cycle_profiles['mass_carrier_gas_out'][desorption_stage_idx]))
        H2O_mass_fraction = (cycle_profiles['mass_H2O_out'][desorption_stage_idx] /
                            (cycle_profiles['mass_CO2_out'][desorption_stage_idx] +
                             cycle_profiles['mass_carrier_gas_out'][desorption_stage_idx]))
        #print(f"H2O mass fraction in product: {H2O_mass_fraction:.6f}") 
        cycle_purity_dry = cycle_purity / H2O_mass_fraction
        print(f"Cycle purity (dry basis): {cycle_purity_dry:.6f}")
        
        mass_CO2_out = cycle_profiles['mass_CO2_out'][desorption_stage_idx] # kg
        CO2_production_rate = mass_CO2_out / cycle_time * 60 * 60 # kg/hr

        thermal_energy_consumption = np.sum(cycle_profiles['thermal_energy_input']) + np.sum(cycle_profiles['thermal_energy_input']) # J
        specific_thermal_energy = thermal_energy_consumption / mass_CO2_out / 1e9 * 1000  # GJ/tCO2

        mechanical_energy = np.sum(cycle_profiles['vacuum_energy_input']) + np.sum(cycle_profiles['fan_energy_input']) # J
        specific_mechanical_energy = mechanical_energy / mass_CO2_out / 1e9 * 1000  # GJ/tCO2

        recovery_rate = (cycle_profiles['mass_CO2_out'][desorption_stage_idx] /
                        sum(cycle_profiles['mass_CO2_in']))
        
     
        productivity = mass_CO2_out / bed_properties["sorbent_mass"] / cycle_time  # kgCO2 / kg_sorbent / s
        daily_productivity = productivity * 86400 * bed_properties["sorbent_mass"] # kgCO2/day
        daily_productivity /= 1000 # tCO2/day
        bed_size_factor = bed_properties["sorbent_mass"] / daily_productivity

        print(f"Cycle {cycle + 1} Results:")
        print(f"  Purity: {cycle_purity:.6f}")
        print(f"  Recovery Rate: {recovery_rate:.6f}")
        print(f"  CO2 Production Rate (kg/hr): {CO2_production_rate:.6e}")
        print(f"  Specific Thermal Energy (GJ/tCO2): {specific_thermal_energy:.6f}")
        print(f"  Specific Mechanical Energy (GJ/tCO2): {specific_mechanical_energy:.6f}")
        print("Other Metrics:")
        print(f"  Daily Productivity (tCO2/day): {daily_productivity:.6e}")
        print(f"  Bed Size Factor: {bed_size_factor:.2e}")
        print(f"  Cycle Error: {cycle_error_value}")
        

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

    return (profiles['temperature'], profiles['outlet_CO2'], profiles['outlet_H2O'], profiles['time'],
            profiles['pressure_inlet'], profiles['pressure_outlet'],
            profiles['adsorbed_CO2'], profiles['adsorbed_H2O'], profiles['wall_temperature'], all_cycle_errors)


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
    n_cycles = 10
    
    simulation_results = run_cycle(n_cycles)
    
    if simulation_results is None:
        print("Simulation failed!")
        return
    
    # Unpack results
    (temperature_profile, outlet_CO2_profile, outlet_H2O_profile, time_profile, 
     pressure_profile_inlet, pressure_profile_outlet, adsorbed_CO2_profile, 
     adsorbed_H2O_profile, wall_temperature_profile, all_cycle_errors) = simulation_results


    # Create visualization
    create_multi_plot(simulation_results, bed_properties)
    create_quick_plot(np.arange(1, len(all_cycle_errors)+1), 
                     np.log10(all_cycle_errors), "Cycle Convergence", "Log₁₀(Cycle Error)")
    
    print("\nSimulation completed successfully!")


if __name__ == "__main__":
    main()