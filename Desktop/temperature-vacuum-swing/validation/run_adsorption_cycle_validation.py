
import numpy as np
from scipy.integrate import solve_ivp
from additional_functions_validation import create_non_uniform_grid, adsorption_isotherm_1, adsorption_isotherm_2, total_mass_balance_error, CO2_mass_balance_error, energy_balance_error, create_plot, create_combined_plot, create_quick_plot, build_jacobian
from scipy import integrate
import time
import matplotlib.pyplot as plt
import tvsa_adsorption_column_validation as column

"""This script sets up the initial conditions and parameters for an adsorption column model simulation.
It defines the column grid, bed properties, inlet and outlet conditions, and initializes the state variables for the simulation.
The components are (1) CO2, (2) H2O, (3) N2 and (4) O2."""


# Define fixed bed properties
def create_fixed_properties():
    bed_properties = {
    #Column dimensions and properties
    "bed_length":0.286,  # Example value for bed length in meters
    "inner_bed_radius": 0.00945,
    "outer_bed_radius": 0.0127,  # Example value for bed diameter in meters
    "bed_voidage": 0.456,  # Example value for bed voidage
    "total_voidage": 0.456+(1-0.456)*0.59,  # Example value for total voidage
    "bed_density": 435,  # Example value for bed density in kg/m^3
    "column_area": 0.00945**2 * np.pi,  # Cross-sectional area of the column
    "sorbent_density": 900, #kg/m3, 55.4 (Haghpanagh et al. 2013)
    "ambient_temperature": 293.15,  # Example value for ambient temperature in Kelvin
    
    # Properties and constants
    "particle_voidage": 0.59,  # Example value for particle voidage
    "tortuosity": 3,  # Example value for tortuosity
    "molecular_diffusivity": 0.605e-5,  # Example value for molecular diffusivity in m²/s
    "particle_diameter": 2e-3,  # Example value for particle diameter in meters
    "wall_density": 2700,  # Example value for wall density in kg/m^3 (Haghpanagh et al. 2013)
    "R": 8.314,  # Universal gas constant in J/(mol*K)
    "sorbent_mass": 0.286 * 0.00945**2 * np.pi* 435,  # Example value for sorbent mass in kg
    "sorbent_heat_capacity": 1040, # J /kg/K, # Example value for solid heat capacity
    "wall_heat_capacity": 902, # J /kg/K, # Example value for wall heat capacity (Haghpanagh et al. 2013)
    "heat_capacity_1": 840 * 44.01 / 1000,  # J / kgK / (kg/mol) J/mol K * kg/m3 Example value for adsorbed phase heat capacity of component 1 (CO2) 
    "heat_capacity_2": 30,  # Example value for adsorbed phase heat capacity of component 2 (H2O)
    "mass_transfer_1": 0.0002,  # Example value for mass transfer coefficient of component 1 (CO2) in s-1
    "mass_transfer_2": 0.002,  # Example value for mass transfer coefficient of component 2 (H2O) in s-1
    "K_z": 0.1, # Example value for axial dispersion coefficient in m²/s
    "h_bed": 50,  # 140,  # Example value for bed heat transfer coefficient in W/m²·K
    "h_wall": 7,  # 20,  # Example value for wall heat transfer coefficient in W
    "MW_1": 44.01,  # Molecular weight of component 1 (CO2) in g/mol
    "MW_2": 18.02,  # Molecular weight of component 2 (H2O) in g/mol
    "MW_3": 28.02,  # Molecular weight of component 3 (N2) in g/mol
    "MW_4": 32.00,  # Molecular weight of component 4 (O2) in g/mol
    "mu": 1.78e-5,  # Example value for feed viscosity in Pa.s
    "isotherm_type_1": "Dual_site Langmuir",
    "isotherm_type_2": "None",  # Example value for isotherm type for component 2 (H2O)

    #Reference values for scaling
    "P_ref": 101325,  # Reference pressure in Pa
    "T_ref": 298.15,  # Reference temperature in K
    "n_ref": 3000,  # Reference adsorbed amount in mol/m^3

    "t_tot": 100,
}
    column_grid = create_non_uniform_grid(bed_properties)

    # Initial values for the column
    P = np.ones(column_grid["num_cells"]) * 101325  # Example pressure vector in Pa
    T = np.ones(column_grid["num_cells"]) * 292  # Example temperature vector in K
    Tw = np.ones(column_grid["num_cells"]) * 292  # Example wall temperature vector in K
    y1 = np.ones(column_grid["num_cells"]) * 0.02  # Example mole fraction of component 1
    y2 = np.ones(column_grid["num_cells"]) * 1e-6 # Example mole fraction of component 2
    y3 = np.ones(column_grid["num_cells"]) * 0.95  # Example mole fraction of component 3
    n1 = adsorption_isotherm_1(P, T, y1, y2, y3, 1e-6, bed_properties=bed_properties, isotherm_type_1=bed_properties["isotherm_type_1"])[0]  # Example concentration vector for component 1 in mol/m^3
    n2 = adsorption_isotherm_2(P, T, y2, bed_properties=bed_properties, isotherm_type=bed_properties["isotherm_type_2"])[0]# Example concentration vector for component 2 in mol/m^3
    F = np.zeros(8)
    E = np.zeros(3)  # Additional variables (e.g., flow rates, mass balances)
    initial_conditions = np.concatenate([P/bed_properties["P_ref"], T/bed_properties["T_ref"], Tw/ bed_properties["T_ref"], y1, y2, y3, n1/bed_properties["n_ref"], n2/bed_properties["n_ref"], F, E])

    #Tolerances
    rtol = 1e-5
    atol_P = 1e-5 * np.ones(len(P))
    atol_T = 1e-4 * np.ones(len(T))
    atol_Tw = 1e-5 * np.ones(len(Tw))
    atol_y1 = 1e-8 * np.ones(len(y1))
    atol_y2 = 1e-8 * np.ones(len(y2))
    atol_y3 = 1e-8 * np.ones(len(y3))
    atol_n1 = 1e-3 * np.ones(len(n1))
    atol_n2 = 1e-3* np.ones(len(n2))
    atol_F = 1e-4 * np.ones(len(F))
    atol_E = 1e-4 * np.ones(len(E))
    atol_array = np.concatenate([atol_P, atol_T, atol_Tw, atol_y1, atol_y2, atol_y3, atol_n1, atol_n2, atol_F, atol_E])
    

    return bed_properties, column_grid, initial_conditions, rtol, atol_array

# Define pressure ramp function for blowdown stage
def pressure_ramp(t, stage, ramp_duration=2.0):
    """
    Ramps pressure down from 101325 Pa to 40000 Pa over the specified duration.
    
    Parameters:
    t (float): Current time in seconds
    stage (str): Stage name ("blowdown" or "cooling")
    ramp_duration (float): Duration of pressure ramp in seconds (default: 2.0)
    
    Returns:
    float: Pressure in Pa
    """

    if stage == "blowdown":
        initial_pressure = 101325  # Pa
        final_pressure = 40000     # Pa

        if t <= ramp_duration:
            # Linear ramp over ramp_duration
            pressure = initial_pressure - (initial_pressure - final_pressure) * (t / ramp_duration)
        else:
        # Constant pressure after ramp
            pressure = final_pressure

    elif stage == "cooling":
        initial_pressure = 40000  # Pa
        final_pressure = 101325     # Pa
        # Use much longer ramp duration for cooling to avoid steep gradients
        cooling_ramp_duration = max(ramp_duration * 10, 50.0)  # At least 50 seconds for cooling
        
        if t <= cooling_ramp_duration:
            # Linear ramp over cooling_ramp_duration
            pressure = initial_pressure + (final_pressure - initial_pressure) * (t / cooling_ramp_duration)
        else:
            pressure = final_pressure

    return pressure

# Define bed inlet values (subject to variation)
def define_boundary_conditions(stage, bed_properties):
    
    if stage == "adsorption":
        left_type = "mass_flow"
        right_type = "pressure"
        column_direction = "forwards"
    elif stage == "blowdown":
        left_type = "closed"
        right_type = "pressure"
        column_direction = "forwards"
    elif stage == "heating":
        left_type = "closed"
        right_type = "pressure"
        column_direction = "forwards"
    elif stage == "cooling":
        left_type = "mass_flow"
        right_type = "pressure"
        column_direction = "forwards"

    def left_velocity():
        if stage == "adsorption":
            velocity = 100 / 60 / 1e6 / bed_properties["column_area"] / bed_properties["bed_voidage"]
            return velocity
        elif stage == "blowdown":
            velocity = 0
            return velocity
        elif stage == "heating":
            velocity = 0
            return velocity
        elif stage == "cooling":
            velocity = 100 / 60 / 1e6 / bed_properties["column_area"] / bed_properties["bed_voidage"]
            return velocity

    def left_pressure():
        if stage == "adsorption":
            # Return a constant pressure function
            def pressure_func(t):
                return 101325
            return pressure_func
        elif stage == "blowdown":
            # Return the pressure ramp function for blowdown
            def pressure_func(t):
                return None
            return pressure_func
        elif stage == "heating":
            # Return a constant pressure function
            def pressure_func(t):
                return None
            return pressure_func
        elif stage == "cooling":
            # Return the pressure ramp function for cooling
            def pressure_func(t):
                return pressure_ramp(t, "cooling", ramp_duration=10.0)
            return pressure_func
    
    def left_temperature():
        temperature = 293.15
        return temperature
    
    def left_gas_composition():
        y1 = 0.15
        y2 = 1e-6
        y3 = 0.84
        return y1, y2, y3

    left_values = {
        "left_type": left_type,
        "velocity": left_velocity(),  # Example value for interstitial velocity in m/s
        "left_volume_flow": 1.6667e-6,  # cm³/min to m³/s
        "left_mass_flow": (0.01 * float(bed_properties["column_area"]) * 1.13),  # Example value for feed mass flow in kg/s
        "left_temperature": left_temperature(),  # Example value for feed temperature in Kelvin
        "left_pressure": left_pressure(),  # Example value for feed pressure in Pa
        "y1_left_value": left_gas_composition()[0],  # Example value for feed mole fraction
        "y2_left_value": left_gas_composition()[1],  # Example value for feed mole fraction
        "y3_left_value": left_gas_composition()[2],  # Example value for feed mole fraction
    }

    def right_pressure():
        if stage == "adsorption":
            # Return a constant pressure function
            def pressure_func(t):
                return 101325
            return pressure_func
        elif stage == "blowdown":
            # Return the pressure ramp function for blowdown
            def pressure_func(t):
                return pressure_ramp(t, "blowdown")
            return pressure_func
        elif stage == "heating":
            # Return a constant pressure function
            def pressure_func(t):
                return 40000
            return pressure_func
        elif stage == "cooling":
            # Return the pressure ramp function for cooling
            def pressure_func(t):
                return pressure_ramp(t, "cooling", ramp_duration=10.0)
            return pressure_func

    def right_temperature():
        temperature = 293.15
        return temperature

    def right_gas_composition():
        y1 = 0.9999
        y2 = 1e-6
        y3 = 1e-6
        return y1, y2, y3

    right_values = {
        "right_type": right_type,
        "right_pressure_func": right_pressure(),  # Function to calculate pressure as function of time
        "right_temperature": right_temperature(),  # Example value for outlet temperature in Kelvin
        "y1_right_value": right_gas_composition()[0],  # Example value for feed mole fraction
        "y2_right_value": right_gas_composition()[1],  # Example value for feed mole fraction
        "y3_right_value": right_gas_composition()[2],  # Example value for feed mole fraction
    }
    
    return left_values, right_values, column_direction, stage


bed_properties, column_grid, initial_conditions, rtol, atol_array = create_fixed_properties()
time_profile = []
temperature_profile = []
pressure_profile = []
outlet_CO2_profile = []
adsorbed_CO2_profile = []
wall_temperature_profile = []

#jacobian = build_jacobian(column_grid["num_cells"])

# Running simulation! ======================================================================================================

def run_stage(left_values, right_values, column_direction, stage, t_span, initial_conditions):
    t0 = time.time()
    def ODE_func(t, results_vector,):
        return column.ODE_calculations(t, results_vector=results_vector, column_grid=column_grid, bed_properties=bed_properties, left_values=left_values, right_values=right_values, column_direction=column_direction, stage=stage)
    output_matrix = solve_ivp(ODE_func, t_span, initial_conditions, method='BDF', rtol=rtol, atol=atol_array)
    t1=time.time()
    total_time = t1 - t0
        
    P_result = output_matrix.y[0:column_grid["num_cells"]] * bed_properties["P_ref"]
    T_result = output_matrix.y[column_grid["num_cells"]:2*column_grid["num_cells"]] * bed_properties["T_ref"]
    Tw_result = output_matrix.y[2*column_grid["num_cells"]:3*column_grid["num_cells"]] * bed_properties["T_ref"]
    y1_result = output_matrix.y[3*column_grid["num_cells"]:4*column_grid["num_cells"]]
    y2_result = output_matrix.y[4*column_grid["num_cells"]:5*column_grid["num_cells"]]
    y3_result = output_matrix.y[5*column_grid["num_cells"]:6*column_grid["num_cells"]]
    n1_result = output_matrix.y[6*column_grid["num_cells"]:7*column_grid["num_cells"]] * bed_properties["n_ref"]
    n2_result = output_matrix.y[7*column_grid["num_cells"]:8*column_grid["num_cells"]] * bed_properties["n_ref"]
    F_result = output_matrix.y[8*column_grid["num_cells"]:8*column_grid["num_cells"]+8]
    E_result = output_matrix.y[8*column_grid["num_cells"]+8:]
    time_array = output_matrix.t

    print(f"Completed {stage} stage")
    print("Mass balance error:", total_mass_balance_error(F_result, P_result, T_result, n1_result, n2_result, time_array, bed_properties, column_grid))
    #print("CO2 mass balance error:", CO2_mass_balance_error(F_result, P_result, T_result, y1_result, n1_result, time_array, bed_properties, column_grid))
    #print("Energy balance error:", energy_balance_error(E_result, T_result, P_result, y1_result, y2_result, y3_result, n1_result, n2_result, Tw_result, time_array, bed_properties, column_grid))
    #print("Total simulation time of stage:", total_time, "seconds")
    print("Duration of stage:", time_array[-1], "seconds")

    # Calculate the exit column values
    (P_walls_result, T_walls_result, Tw_walls_result,
    y1_walls_result, y2_walls_result, y3_walls_result,
    v_walls_result) = column.final_wall_values(column_grid, bed_properties, left_values, right_values, output_matrix)

    # Adjust time array to account for previous stages
    if len(time_profile) == 0:
        time_offset = 0
    else:
        time_offset = time_profile[-1]
        
    print("Start time:", time_offset)
    adjusted_time_array = time_array + time_offset
    print("length of simulation", adjusted_time_array[-1])
    print("-------------------------------")

    # Extend the profiles instead of appending arrays
    time_profile.extend(adjusted_time_array)
    temperature_profile.extend(T_result[9])
    outlet_CO2_profile.extend(y1_walls_result[-1])
    pressure_profile.extend(P_walls_result[0])
    adsorbed_CO2_profile.extend(n1_result[9])
    wall_temperature_profile.extend(Tw_walls_result[9])

    #Extract final values as initial conditions
    P_final = P_result[:,-1]
    T_final = T_result[:,-1]
    Tw_final = Tw_result[:,-1]
    y1_final = y1_result[:,-1]
    y2_final = y2_result[:,-1]
    y3_final = y3_result[:,-1]
    n1_final = n1_result[:,-1]
    n2_final = n2_result[:,-1]
    F_final = np.zeros(8)
    E_final = np.zeros(3)

    final_conditions = np.concatenate([P_final/bed_properties["P_ref"], T_final/bed_properties["T_ref"], Tw_final/ bed_properties["T_ref"], y1_final, y2_final, y3_final, n1_final/bed_properties["n_ref"], n2_final/bed_properties["n_ref"], F_final, E_final])

    #create_plot(time_array, T_result, "Temperature evolution", "Temperature")
    #create_plot(time_array, P_result, "Pressure evolution", "Pressure")
    #create_plot(time_array, y1_result, "CO2 mole fraction evolution", "Mole fraction")
    #create_plot(time_array, n1_result, "CO2 adsorbed amount evolution", "Adsorbed amount")

    return final_conditions, time_profile, temperature_profile, outlet_CO2_profile,  pressure_profile, adsorbed_CO2_profile, wall_temperature_profile

def run_cycle(n_cycles):
    global time_profile, temperature_profile, pressure_profile, outlet_CO2_profile, adsorbed_CO2_profile, wall_temperature_profile
    current_initial_conditions = initial_conditions  # Start with global initial conditions
    
    for cycle in range(n_cycles):
        print("========================================")
        print(f"Starting cycle {cycle + 1} of {n_cycles}")

        # Clear the global profiles at the start of each cycle
        time_profile.clear()
        temperature_profile.clear()
        pressure_profile.clear()
        outlet_CO2_profile.clear()
        adsorbed_CO2_profile.clear()
        wall_temperature_profile.clear()
        
        # Stage 1 of cycle - Adsorption ===========================================================================================
        left_values, right_values, column_direction, stage = define_boundary_conditions("adsorption", bed_properties)
        t_span = [0, 1000]  # Time span for the ODE solver
        final_conditions_adsorption, _, _, _, _, _, _ = run_stage(left_values, right_values, column_direction, stage, t_span, current_initial_conditions)
    
        # Stage 2 of cycle - Blowdown ===============================================================================================
        # Define boundary conditions for blowdown stage
        left_values_blowdown, right_values_blowdown, column_direction_blowdown, stage = define_boundary_conditions("blowdown", bed_properties)
        t_span_blowdown = [0, 100]  # Time span for the ODE solver
        final_conditions_blowdown, _, _, _, _, _, _ = run_stage(left_values_blowdown, right_values_blowdown, column_direction_blowdown, stage, t_span_blowdown, final_conditions_adsorption)
    
        # Stage 3 of cycle - Heating ===============================================================================================
        # Define boundary conditions for heating stage
        left_values_heating, right_values_heating, column_direction_heating, stage = define_boundary_conditions("heating", bed_properties)
        final_conditions_blowdown[2*column_grid["num_cells"]:3*column_grid["num_cells"]] = 400/bed_properties["T_ref"]  # Set wall temperature to 400 K
        t_span_heating = [0, 500]  # Time span for the ODE solver
        final_conditions_heating, _, _, _, _, _, _ = run_stage(left_values_heating, right_values_heating, column_direction_heating, stage, t_span_heating, final_conditions_blowdown)
    
        # Stage 4 of cycle - Cooling ===============================================================================================
        left_values_cooling, right_values_cooling, column_direction_cooling, stage = define_boundary_conditions("cooling", bed_properties)
        t_span_cooling = [0, 2000]  # Time span for the ODE solver
        final_conditions_cooling, _, _, _, _, _, _ = run_stage(left_values_cooling, right_values_cooling, column_direction_cooling, stage, t_span_cooling, final_conditions_heating)
        
        # Update initial conditions for next cycle
        current_initial_conditions = final_conditions_cooling
        
        print(f"Completed cycle {cycle + 1}")
    
    return final_conditions_cooling, temperature_profile, outlet_CO2_profile, time_profile, pressure_profile, adsorbed_CO2_profile, wall_temperature_profile

def main():
    # Run multiple cycles to reach steady state

    bed_properties, column_grid, initial_conditions, rtol, atol_array = create_fixed_properties()

    final_conditions, final_temperature_profile, final_outlet_CO2_profile, final_time_profile, final_pressure_profile, final_adsorbed_CO2_profile, final_wall_temperature_profile = run_cycle(5)
      # Run 5 cycles

    # Create plots for the final cycle
    create_quick_plot(final_time_profile, final_temperature_profile, "Temperature at column midpoint (Final Cycle)", "Temperature (K)")
    create_quick_plot(final_time_profile, final_pressure_profile, "Pressure at column inlet (Final Cycle)", "Pressure (Pa)")
    create_quick_plot(final_time_profile, final_outlet_CO2_profile, "Outlet CO2 concentration (Final Cycle)", "Mole Fraction")
    create_quick_plot(final_time_profile, final_adsorbed_CO2_profile, "Adsorbed CO2 at column midpoint (Final Cycle)", "Concentration (mol/m³)")
    create_quick_plot(final_time_profile, final_wall_temperature_profile, "Wall Temperature at column midpoint (Final Cycle)", "Temperature (K)")

    create_combined_plot(final_time_profile, final_temperature_profile, final_outlet_CO2_profile, final_pressure_profile, final_adsorbed_CO2_profile, final_wall_temperature_profile)

if __name__ == "__main__":
    main()
