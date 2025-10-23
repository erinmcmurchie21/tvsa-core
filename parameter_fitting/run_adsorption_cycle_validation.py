import numpy as np
from scipy.integrate import solve_ivp
from additional_functions_validation import (
    create_non_uniform_grid,
    adsorption_isotherm_1,
    adsorption_isotherm_2,
    total_mass_balance_error,
    CO2_mass_balance_error,
    energy_balance_error,
    create_combined_plot,
)
import time
import tvsa_adsorption_column_validation as column

"""This script sets up the initial conditions and parameters for an adsorption column model simulation.
It defines the column grid, bed properties, left and right conditions, and initializes the state variables for the simulation.
The components are (1) CO2, (2) H2O, (3) N2 and (4) O2."""


# Define fixed bed properties
def create_fixed_properties():
    bed_properties = {
        # Column dimensions and properties
        "bed_length": 0.286,  # Example value for bed length in meters
        "inner_bed_radius": 0.00945,
        "outer_bed_radius": 0.0127,  # Example value for bed diameter in meters
        "bed_voidage": 0.456,  # Example value for bed voidage
        "total_voidage": 0.456 + (1 - 0.456) * 0.59,  # Example value for total voidage
        "bed_density": 435,  # Example value for bed density in kg/m^3
        "column_area": 0.00945**2 * np.pi,  # Cross-sectional area of the column
        "sorbent_density": 900,  # kg/m3, 55.4 (Haghpanagh et al. 2013)
        "ambient_temperature": 293.15,  # Example value for ambient temperature in Kelvin
        # Properties and constants
        "particle_voidage": 0.59,  # Example value for particle voidage
        "tortuosity": 3,  # Example value for tortuosity
        "molecular_diffusivity": 0.605e-5,  # Example value for molecular diffusivity in m²/s
        "particle_diameter": 3e-3,  # Example value for particle diameter in meters
        "wall_density": 2700,  # Example value for wall density in kg/m^3 (Haghpanagh et al. 2013)
        "R": 8.314,  # Universal gas constant in J/(mol*K)
        "sorbent_mass": 0.286
        * 0.00945**2
        * np.pi
        * 435,  # Example value for sorbent mass in kg
        "sorbent_heat_capacity": 1040,  # J /kg/K, # Example value for solid heat capacity
        "wall_heat_capacity": 902,  # J /kg/K, # Example value for wall heat capacity (Haghpanagh et al. 2013)
        "heat_capacity_1": 840
        * 44.01
        / 1000,  # J / kgK / (kg/mol) J/mol K * kg/m3 Example value for adsorbed phase heat capacity of component 1 (CO2)
        "heat_capacity_2": 30,  # Example value for adsorbed phase heat capacity of component 2 (H2O)
        "mass_transfer_1": 0.0002,  # Example value for mass transfer coefficient of component 1 (CO2) in s-1
        "mass_transfer_2": 0,  # Example value for mass transfer coefficient of component 2 (H2O) in s-1
        "mass_transfer_3": 0,  # Example value for mass transfer coefficient of component 3 (N2) in s-1
        "K_z": 1,  # Example value for axial dispersion coefficient in m²/s
        "K_wall": 205,  # Example value for wall thermal conductivity in W/m·K
        "h_bed": 50,  # 140,  # Example value for bed heat transfer coefficient in W/m²·K
        "h_wall": 7,  # 20,  # Example value for wall heat transfer coefficient in W
        "MW_1": 44.01,  # Molecular weight of component 1 (CO2) in g/mol
        "MW_2": 18.02,  # Molecular weight of component 2 (H2O) in g/mol
        "MW_3": 28.02,  # Molecular weight of component 3 (N2) in g/mol
        "MW_4": 32.00,  # Molecular weight of component 4 (O2) in g/mol
        "mu": 1.72e-5,  # Example value for feed viscosity in Pa.s
        "isotherm_type_1": "Dual_site Langmuir",
        "isotherm_type_2": "None",  # Example value for isotherm type for component 2 (H2O)
        "isotherm_type_3": "None",  # Example value for isotherm type for component 3 (N2)
        # Reference values for scaling
        "P_ref": 101325,  # Reference pressure in Pa
        "T_ref": 298.15,  # Reference temperature in K
        "n_ref": 3000,  # Reference adsorbed amount in mol/m^3
        "t_tot": 100,
        "compressor_efficiency": 0.7,
        "fan_efficiency": 0.7,
        "k": 1.4,
        "ambient_pressure": 101325,
        "ambient_temperature": 293.15,
        "Cp_water": 4184,  # J/kg/K
        "Cp_steam": 2010,  # J/kg/K
        "latent_heat_water": 2260e3,  # J/kg
        "vaporization_energy": 2257e3,  # J/kg
        "steam_temperature": 373.15,  # K
        "rho_gas_initial": 1.13,  # Initial gas density in kg
    }
    column_grid = create_non_uniform_grid(bed_properties)
    
    # Initial values for the column
    P = np.ones(column_grid["num_cells"]) * 101325  # Example pressure vector in Pa
    T = np.ones(column_grid["num_cells"]) * 291.5  # Example temperature vector in K
    Tw = (
        np.ones(column_grid["num_cells"]) * 291.5
    )  # Example wall temperature vector in K
    y1 = np.ones(column_grid["num_cells"]) * 0.0  # Example mole fraction of component 1
    y2 = (
        np.ones(column_grid["num_cells"]) * 1e-6
    )  # Example mole fraction of component 2
    y3 = (
        np.ones(column_grid["num_cells"]) * 0.95
    )  # Example mole fraction of component 3
    n1 = adsorption_isotherm_1(
        P,
        T,
        y1,
        y2,
        y3,
        1e-6,
        1e-6,
        bed_properties=bed_properties,
        isotherm_type_1=bed_properties["isotherm_type_1"],
    )[0]  # Example concentration vector for component 1 in mol/m^3
    n2 = adsorption_isotherm_2(
        P,
        T,
        y2,
        bed_properties=bed_properties,
        isotherm_type_2=bed_properties["isotherm_type_2"],
    )[0]  # Example concentration vector for component 2 in mol/m^3
    n3 = np.zeros(column_grid["num_cells"])  # Example concentration vector for component 3 in mol/m^3  
    F = np.zeros(8)
    E = np.zeros(7)  # Additional variables (e.g., flow rates, mass balances)
    initial_conditions = np.concatenate(
        [
            P / bed_properties["P_ref"],
            T / bed_properties["T_ref"],
            Tw / bed_properties["T_ref"],
            y1,
            y2,
            y3,
            n1 / bed_properties["n_ref"],
            n2 / bed_properties["n_ref"],
            n3 / bed_properties["n_ref"],
            F,
            E,
        ]
    )

    rtol = 1e-4

    atol_P = 1e-4 * np.ones(len(P))
    atol_T = 1e-4 * np.ones(len(T))
    atol_Tw = 1e-4 * np.ones(len(Tw))
    atol_y1 = 1e-6 * np.ones(len(y1))
    atol_y2 = 1e-6 * np.ones(len(y2))
    atol_y3 = 1e-6 * np.ones(len(y3))
    atol_n1 = 1e-3 * np.ones(len(n1))
    atol_n2 = 1e-3 * np.ones(len(n2))
    atol_n3 = 1e-3 * np.ones(len(n3))
    atol_F = 1e-4 * np.ones(len(F))
    atol_E = 1e-4 * np.ones(len(E))
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
            atol_n3,
            atol_F,
            atol_E,
        ]
    )

    return bed_properties, column_grid, initial_conditions, rtol, atol_array


# Define bed left values (subject to variation)
def define_boundary_conditions(bed_properties):
    left_values = {
        "left_type": "mass_flow",
        "velocity": 100
        / 60
        / 1e6
        / bed_properties["column_area"]
        / bed_properties["bed_voidage"]
        ,  # Example value for interstitial velocity in m/s
        "rho_gas": 1.13,  # Example value for feed density in kg/m^3
        "feed_volume_flow": 1.6667e-6,  # cm³/min to m³/s
        "feed_mass_flow": (
            0.01 * float(bed_properties["column_area"]) * 1.13
        ),  # Example value for feed mass flow in kg/s
        "left_temperature": 293.15,  # Example value for feed temperature in Kelvin
        "left_pressure": 100000,  # Example value for feed pressure in Pa
        "y1_left_value": 0.999999,  # Example value for feed mole fraction
        "y2_left_value": 1e-6,  # Example value for feed mole fraction
        "y3_left_value": 1e-6,  # Example value for feed mole fraction
    }
    # Define bed right values (subject to variation)
    right_values = {
        "right_type": "pressure",
        "right_pressure": 101325,  # Example value for right pressure in Pa
        "right_temperature": 293.15,  # Example value for right temperature in Kelvin
    }

    column_direction = "forwards"
    return left_values, right_values, column_direction


# Running simulation! ======================================================================================================

if __name__ == "__main__":
    print("Starting simulation...")

    # Step 1: Run first 10 seconds
    time_start = time.time()
    bed_properties, column_grid, initial_conditions, rtol, atol_array = create_fixed_properties()
    left_values, right_values, column_direction = define_boundary_conditions(bed_properties)
    t_span_1 = [0, 10]
    max_step = 0.1
    first_step = 1e-3

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

    output_matrix_1 = solve_ivp(
        ODE_func,
        t_span_1,
        initial_conditions,
        method="BDF",
        rtol=rtol,
        atol=atol_array,
        max_step=0.1,
        first_step=1e-3,
    )
    time_end = time.time()
    total_simulation_time = time_end - time_start
    print("Total simulation time:", total_simulation_time, "seconds")
    # Step 2: Use final state as initial condition for next simulation
    initial_conditions_2 = output_matrix_1.y[:, -1]
    t_span_2 = [10, 3000]

    output_matrix_2 = solve_ivp(
        ODE_func,
        t_span_2,
        initial_conditions_2,
        method="BDF",
        rtol=rtol,
        atol=atol_array,
        max_step=10,
        first_step=1e-3,
    )

    # Step 3: Stitch results together
    time_array = np.concatenate([output_matrix_1.t, output_matrix_2.t])
    Y = np.concatenate([output_matrix_1.y, output_matrix_2.y], axis=1)

    # Now extract results as before, but from stitched Y and time
    P_result = Y[0 : column_grid["num_cells"]] * bed_properties["P_ref"]
    T_result = Y[column_grid["num_cells"] : 2 * column_grid["num_cells"]] * bed_properties["T_ref"]
    Tw_result = Y[2 * column_grid["num_cells"] : 3 * column_grid["num_cells"]] * bed_properties["T_ref"]
    y1_result = Y[3 * column_grid["num_cells"] : 4 * column_grid["num_cells"]]
    y2_result = Y[4 * column_grid["num_cells"] : 5 * column_grid["num_cells"]]
    y3_result = Y[5 * column_grid["num_cells"] : 6 * column_grid["num_cells"]]
    n1_result = Y[6 * column_grid["num_cells"] : 7 * column_grid["num_cells"]] * bed_properties["n_ref"]
    n2_result = Y[7 * column_grid["num_cells"] : 8 * column_grid["num_cells"]] * bed_properties["n_ref"]
    n3_result = Y[8 * column_grid["num_cells"] : 9 * column_grid["num_cells"]] * bed_properties["n_ref"]
    F_result = Y[8 * column_grid["num_cells"] : 8 * column_grid["num_cells"] + 8]
    E_result = Y[8 * column_grid["num_cells"] + 8 :]
    
    print("Simulation complete!")

    # Continue with plotting and analysis as before
    create_combined_plot(time_array, T_result, P_result, y1_result, n1_result, y1_result, F_result, bed_properties)

# ...existing code...