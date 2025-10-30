import numpy as np
import matplotlib.pyplot as plt
from additional_functions_multistage import (
    create_non_uniform_grid, mole_fraction_to_relative_humidity
)


def create_fixed_properties():
    """
    Define fixed bed properties, grid configuration, and initial conditions.

    Returns:
        tuple: (bed_properties, column_grid, initial_conditions, rtol, atol_array)
    """
    # Column dimensions and physical properties
    bed_properties = {


        # Geometry - FROM PAPER TABLE 1
        "bed_length": 1.2,  # Bed length [m]
        "inner_bed_radius": 0.015,  # Inner radius [m] 
        "outer_bed_radius": 0.016,  # Outer radius [m] 
        "column_area": 0.015**2 * np.pi,  # Cross-sectional area [m²]
        
        # Porosity and density - FROM PAPER TABLE 1 & CALCULATIONS
        "bed_voidage": 0.3475,  # Bed voidage [-] 
        "particle_voidage": 0.5401,  # Particle voidage [-] 
        "total_voidage": 0.6999,  # Total voidage [-]
        "bed_density": 708,  # Bed density [kg/m³]
        "sorbent_density": 1085,  # Particle density [kg/m³]
        "material_density": 2359,  # 13X material density [kg/m³] - ADDED from paper
        "wall_density": 7800,  # Wall density [kg/m³] - NOT IN PAPER, REASONABLE for steel
        
        # Transport properties
        "tortuosity": 3,  # Tortuosity factor [-] - NOT IN PAPER, KEEP ORIGINAL
        "molecular_diffusivity": 1.605e-5,  # Molecular diffusivity [m²/s] - NOT IN PAPER
        "particle_diameter": 0.002,  # Particle diameter [m]
        "K_z": 0.09,  # Axial dispersion coefficient [m²/s]
        "K_wall": 50,  # Wall thermal conductivity [W/m·K]
        "mu": 1.78e-5,  # Gas viscosity [Pa·s]
        
        # Mass transfer coefficients - FROM PAPER TABLE 1
        "mass_transfer_1": 0.15,  # CO2 mass transfer coeff [s⁻¹] - CORRECTED from 0.0002
        "mass_transfer_2": 0,  # N2 mass transfer coeff [s⁻¹] - ADDED from paper
        "mass_transfer_3": 1.00,  # H2O mass transfer coeff [s⁻¹] - NOT USED IN PAPER
        # Note: H2O not modeled in paper (dried upstream), so no k_H2O given
        
        # Heat transfer properties - FROM PAPER TABLE 1
        "h_bed": 20,  # Bed heat transfer coeff [W/m²·K] - CORRECTED (wall/bed in paper)
        "h_wall": 100,  # Wall heat transfer coeff [W/m²·K] - CORRECTED (fluid/wall in paper)
        "sorbent_heat_capacity": 920,  # Solid heat capacity [J/kg·K] - CORRECTED from 2070
        "wall_heat_capacity": 513,  # Wall heat capacity [J/kg·K] - KEEP (not clear in paper)
        "wall_heat_capacity_volumetric": 4e6,  # Wall heat capacity [J/(K·m³)] - FROM PAPER
        "heat_capacity_1": 42.46,  # CO2 adsorbed phase Cp [J/mol·K] - NOT IN PAPER
        "heat_capacity_2": 73.1,  # H2O adsorbed phase Cp [J/mol·K] - NOT IN PAPER
        "heat_capacity_3": 29.1,  # N2 adsorbed phase Cp [J/mol·K] - NOT IN PAPER
        
        # Molecular weights - STANDARD VALUES (correct in original)
        "MW_1": 44.01,  # CO2 [g/mol]
        "MW_2": 18.02,  # H2O [g/mol]
        "MW_3": 28.02,  # N2 [g/mol]
        "MW_4": 32.00,  # O2 [g/mol]
        
        # Thermodynamic properties
        "R": 8.314,  # Universal gas constant [J/mol·K]
        "k": 1.4,  # Heat capacity ratio [-] - NOT IN PAPER
        "ambient_temperature": 300,  # Ambient temperature [K] - NOT IN PAPER
        "ambient_pressure": 101325,  # Ambient pressure [Pa] - CORRECTED to 1.3 bar from paper
        
        # Adsorption isotherms - FROM PAPER TABLE 1
        "isotherm_type_1": "BinarySips",  # CO2 isotherm type - CORRECTED from ModifiedToth
        "isotherm_type_2": "None",  
        "isotherm_type_3": "BinarySips",  # N2 isotherm type - ADDED
        # Reference values for scaling (dimensionless variables)
        "P_ref": 101325,  # Reference pressure [Pa] - NOT IN PAPER
        "T_ref": 298.15,  # Reference temperature [K] - NOT IN PAPER
        "n_ref": 3000,  # Reference adsorbed amount [mol/m³] - NOT IN PAPER
        
        # Water properties - NOT USED IN PAPER (dried upstream)
        "Cp_water": 4181.3,  # Heat capacity of water [J/kg·K]
        "Cp_steam": 2010,  # Heat capacity of steam [J/kg·K]
        "vaporization_energy": 2257e3,  # Latent heat of vaporization for water [J/kg]
        
        # Calculated properties - NEED TO RECALCULATE with new dimensions
        "sorbent_mass": 1.2 * 0.015**2 * np.pi * 708,  # [kg] - UPDATED
        "sorbent_volume": 1.2 * 0.015**2 * np.pi * (1 - 0.3475),  # [m³] - UPDATED
        "bed_volume": 1.2 * 0.015**2 * np.pi,  # [m³] - UPDATED
        
        # Efficiencies
        "compressor_efficiency": 0.75,  # Compressor efficiency - NOT IN PAPER
        "fan_efficiency": 0.5,  # Fan efficiency - NOT IN PAPER

         # Optimisation parameters - FROM PAPER (Table 2, Run 1 for Cycle D as example)
        "outside_temperature": 300,  # Ambient temperature for heat loss [K] - NOT IN PAPER
        "desorption_temperature": 420,  # Desorption/heating temperature [K] - CORRECTED
        "cooling_temperature": 300,  # Cooling temperature [K] - ADDED from paper
        "vacuum_pressure": 101325,  # Vacuum pressure [Pa]
        "pressurisation_pressure": 102000,  # Pressurisation pressure [Pa] - 1.3 bar from paper

        "adsorption_time": 300,  # Adsorption time [s] - Example from Table 2, Run 1
        "desorption_time": 2000,  # Desorption time [s] - NOT USED IN PAPER
        "cooling_time": 1450,  # Cooling time [s] - Example from Table 2
        "pressurisation_time": 50,  # Pressurisation time [s] - NOT USED IN PAPER
        
        
        # Feed conditions - FROM PAPER TABLE 1
        "feed_velocity": 0.5*0.3475,  # Superficial feed velocity [m/s]
        "feed_temperature": 303,  # Feed temperature [K]
        "feed_pressure": 130000,  # Feed pressure [Pa] - 1.3 bar from paper
        "feed_flow_rate": 3.5e-4,  # Feed flow rate [m³/s] - FROM PAPER TABLE 1
        "steam_velocity": 0.025,  # Superficial steam velocity [m/s] - NOT IN PAPER
        "steam_temperature": 368.15,  # Steam temperature [K] - NOT USED IN PAPER
        
        # Feed composition - FROM PAPER (flue gas, not ambient air!)
        "feed_composition": {
            "y1": 0.12,  # CO2 
            "y2": 0.00,  # H2O 
            "y3": 0.88,  # N2 
        },
        "steam_composition": {
            "y1": 1e-6,
            "y2": 1 - 2e-6,
            "y3": 1e-6,
        },
        "stage_config": {
            
            "desorption": {
                "left": "closed",
                "right": "pressure",
                "direction": "forwards",
                "duration": 2100,
            },
            "cooling": {
                "left": "closed",
                "right": "closed",
                "direction": "forwards",
                "duration": 1450,
            },
            "pressurisation": {
                "left": "inlet_pressure",
                "right": "closed",
                "direction": "forwards",
                "duration": 50,
            },
            "adsorption": {
                "left": "mass_flow",
                "right": "pressure",
                "direction": "forwards",
                "duration": 1200,
            },
        },
    }

    # Create spatial discretization grid
    column_grid = create_non_uniform_grid(bed_properties)

    # Initialize state variables
    num_cells = column_grid["num_cells"]

    # Initial conditions: ambient pressure, temperature, and composition
    y2_feed = 1e-6
    P_init = np.ones(num_cells) * 130000  # Pressure [Pa]
    T_init = np.ones(num_cells) * 303  # Gas temperature [K]
    Tw_init = np.ones(num_cells) * 303  # Wall temperature [K]
    y1_init = np.ones(num_cells) * 0.1  # CO2 mole fraction
    y2_init = np.ones(num_cells) * y2_feed  # H2O mole fraction
    y3_init = np.ones(num_cells) * 0.88  # N2 mole fraction

    # Calculate initial adsorbed amounts from equilibrium isotherms
    n1_init = adsorption_isotherm_1(
        P_init,
        T_init,
        y1_init,
        y2_init,
        y3_init,
        0.1,
        y2_feed,
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
    n3_init = adsorption_isotherm_3(
        P_init,
        T_init,
        y1_init,
        y2_init,
        y3_init,
        0.1,
        y2_feed,
        bed_properties=bed_properties,
        isotherm_type_3=bed_properties["isotherm_type_3"],
    )[0]

    # Additional state variables (flow rates and balance errors)
    F_init = np.zeros(8)  # Flow rate variables
    E_init = np.zeros(7)  # Energy balance variables

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
            n3_init / bed_properties["n_ref"],
            F_init,
            E_init,  # Additional variables
        ]
    )

    # Solver tolerance settings
    rtol = 1e-5  # Relative tolerance

    # Absolute tolerances for different variable types
    atol_P = 1e-5 * np.ones(len(P_init))  # Pressure
    atol_T = 1e-5 * np.ones(len(T_init))  # Temperature
    atol_Tw = 1e-5 * np.ones(len(Tw_init))  # Wall temperature
    atol_y1 = 1e-9 * np.ones(len(y1_init))  # CO2 mole fraction
    atol_y2 = 1e-9 * np.ones(len(y2_init))  # H2O mole fraction
    atol_y3 = 1e-9 * np.ones(len(y3_init))  # N2 mole fraction
    atol_n1 = 1e-4 * np.ones(len(n1_init))  # CO2 adsorbed amount
    atol_n2 = 1e-4 * np.ones(len(n2_init))  # H2O adsorbed amount
    atol_n3 = 1e-4 * np.ones(len(n3_init))  # N2 adsorbed amount
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
            atol_n3,
            atol_F,
            atol_E,
        ]
    )

    return bed_properties, column_grid, initial_conditions, rtol, atol_array


def adsorption_isotherm_1(
    pressure, temperature, y1, y2, y3, n1, n2, bed_properties, isotherm_type_1="WADST"
):
    """
    Toth isotherm for CO₂ on solid sorbent.

    Returns:
        equilibrium_loading (mol/m³ of solid)
        adsorption_heat_1 (J/mol)
    """
    R = 8.314  # J/(mol·K)
    bed_density = bed_properties["bed_density"]  # kg/m³
    ε = bed_properties["bed_voidage"]  # bed void fraction

    if isotherm_type_1 == "BinarySips":
        T_0 = 298.15  # reference temperature [K]
        X_CO2 = -0.61684  # [-]
        n_inf = 7.268 * np.exp( X_CO2 * (temperature / T_0 - 1 )) # mol/kg Sips isotherm constant
        Qb_CO2 = 28389  # J/mol
        b_CO2 = 1.129e-4 * np.exp(Qb_CO2 / (R * temperature)) #  -1
        p_CO2 = pressure * y1 / 1e5 # bar
        alpha_CO2 = 0.72378 # [-]
        c_CO2 = 0.42456 + alpha_CO2 * (temperature / T_0 - 1 ) # [-]
        Qb_N2 = 18474  # J/mol
        b_N2 = 5.847e-5 * np.exp(Qb_N2 / (R * temperature))
        p_N2 = y3 * pressure / 1e5 # bar
        alpha_N2 = 0 # [-]
        c_N2 = 0.98624 + alpha_N2 * (temperature / T_0 - 1 ) # [-]
        load_kg = n_inf * ( (b_CO2 * p_CO2) ** (c_CO2) ) / (1 + (b_CO2 * p_CO2) ** (c_CO2) + (b_N2 * p_N2) ** (c_N2) )  # mol/kg
        load_m3 = load_kg * bed_density   # mol/m³
        ΔH = -37000  # J/mol
    
    return load_m3, ΔH


def adsorption_isotherm_2(
    pressure, temperature, y2, bed_properties, isotherm_type="None"
):
    """
    GAB isotherm for H₂O.

    Returns:
        equilibrium_loading (mol/m³ of solid)
        adsorption_heat_2 (J/mol)
    """
    bed_density = bed_properties["bed_density"]  # kg/m³
    ε = bed_properties["bed_voidage"]  # bed void fraction

    if isotherm_type == "None":
        load_m3 = 0 * pressure

    ΔH = -49000  # J/mol

    return load_m3, ΔH

def adsorption_isotherm_3(
    pressure, temperature, y1, y2, y3, n1, n2, bed_properties, isotherm_type_3="BinarySips"
):
    """
    Toth isotherm for CO₂ on solid sorbent.

    Returns:
        equilibrium_loading (mol/m³ of solid)
        adsorption_heat_1 (J/mol)
    """
    R = 8.314  # J/(mol·K)
    bed_density = bed_properties["bed_density"]  # kg/m³
    ε = bed_properties["bed_voidage"]  # bed void fraction

    if isotherm_type_3 == "BinarySips":
        T_0 = 298.15  # reference temperature [K]
        X_N2 = 0
        n_inf = 4.051 * np.exp( X_N2 * (temperature / T_0 - 1 )) # Sips isotherm constant
        Qb_CO2 = 28389  # J/mol
        b_CO2 = 1.129e-4 * np.exp(Qb_CO2 / (R * temperature)) #  bar-1
        p_CO2 = y1 * pressure / 1e5 #bar
        alpha_CO2 = 0.72378
        c_CO2 = 0.42456 + alpha_CO2 * (temperature / T_0 - 1 )
        Qb_N2 = 18474  # J/mol
        b_N2 = 5.847e-5 * np.exp(Qb_N2 / (R * temperature))  #  -1
        p_N2 = y3 * pressure / 1e5
        alpha_N2 = 0
        c_N2 = 0.98624 + alpha_N2 * (temperature / T_0 - 1 )
        load_kg = n_inf * ( (b_N2 * p_N2) ** (c_N2) ) / (1 + (b_CO2 * p_CO2) ** (c_CO2) + (b_N2 * p_N2) ** (c_N2) )  # mol/kg
        load_m3 = load_kg * bed_density   # mol/m³
        ΔH = -18500  # J/mol

    return load_m3, ΔH

def create_multi_plot(profiles, bed_properties):
    """
    Create a grid of subplots for multiple time series.

    Parameters
    ----------
    plots : list of tuples
        Each tuple should be (x, y, title, ylabel)
    ncols : int, optional
        Number of columns in the grid (default = 3)
    figsize : tuple, optional
        Size of the whole figure
    """
    # Import data

    time_temp = np.loadtxt(
        "multistage_cycles/profiles/LJ_profiles/temperature.csv",
        delimiter=",",
        usecols=0,
    )  # First column
    temp = np.loadtxt(
        "multistage_cycles/profiles/LJ_profiles/temperature.csv",
        delimiter=",",
        usecols=1,
    )  # Second column

    time_yCO2 = np.loadtxt(
        "multistage_cycles/profiles/LJ_profiles/yCO2.csv",
        delimiter=",",
        usecols=0,
    )  # First column
    yCO2 = np.loadtxt(
        "multistage_cycles/profiles/LJ_profiles/yCO2.csv",
        delimiter=",",
        usecols=1,
    )  # Second column
      # Second column

    time = profiles["time"]
    T_gas = np.array(profiles["temperature"])
    T_wall = np.array(profiles["wall_temperature"])
    P_inlet = np.array(profiles["pressure_inlet"])
    P_outlet = np.array(profiles["pressure_outlet"])
    outlet_CO2 = np.array(profiles["outlet_CO2"])
    outlet_H2O = np.array(profiles["outlet_H2O"])
    adsorbed_CO2 = np.array(profiles["adsorbed_CO2"])
    adsorbed_H2O = np.array(profiles["adsorbed_H2O"])
    outlet_N2 = np.array(profiles["outlet_N2"])
    # outlet_O2 = np.array(profiles["outlet_O2"])
    # equilibrium_CO2 = np.array(profiles["equilibrium_CO2"])

    bed_density = bed_properties["bed_density"]
    bed_voidage = bed_properties["bed_voidage"]

    adsorbed_CO2 = np.array(adsorbed_CO2) / (bed_density )
    adsorbed_H2O = np.array(adsorbed_H2O) / (bed_density )
    relative_humidity = mole_fraction_to_relative_humidity(outlet_H2O, P_outlet, T_gas)

    figsize = (15, 8)
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.ravel()

    # 1. Temperature overlay (gas and wall)
    ax = axes[0]
    ax.plot(time, T_gas, label="Gas Temp (mid)", color="tab:blue", linestyle="None", marker="o", markersize=2)
    ax.plot(time, T_wall, label="Wall Temp (mid)", color="tab:orange", linestyle="None", marker="o", markersize=2)
    ax.plot(
        time_temp,
        temp,
        color="black",
        linestyle="--",
        alpha=0.7,
    )
    ax.set_title("Temperature Profiles")
    ax.set_ylabel("Temperature (K)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Pressure overlay (inlet and outlet)
    ax = axes[1]
    ax.plot(time, P_inlet, label="Inlet Pressure", color="tab:green", linestyle="None", marker="o", markersize=2)
    ax.plot(time, P_outlet, label="Outlet Pressure", color="tab:red", linestyle="None", marker="o", markersize=2)
    # ax.plot(
    #     time_6,
    #     pressure,
    #     color="black",
    #     linestyle="--",
    #     alpha=0.7,
    # )
    ax.set_title("Pressure Profiles")
    ax.set_ylabel("Pressure (Pa)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. CO2 gas phase (outlet and mid)
    ax = axes[2]
    ax.plot(time, outlet_CO2, label="CO2 Outlet", color="tab:purple", linestyle="None", marker="o", markersize=2)
    ax.plot(time_yCO2, yCO2, color="black", linestyle="--", alpha=0.7)
    ax.set_title("CO2 Gas Phase")
    ax.set_ylabel("CO2 Mole Fraction")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. CO2 adsorbed
    ax = axes[3]
    ax.plot(time, adsorbed_CO2, label="CO2 Adsorbed (mid)", color="tab:blue", linestyle="None", marker="o", markersize=2)
    # ax.plot(
    #      time,
    #      qCO2,
    #      color="black",
    #      linestyle="--",
    #      alpha=0.7,
    # )
    ax.set_title("CO2 Adsorbed")
    ax.set_ylabel("CO2 Loading (mol/kg)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Relative Humidity
    ax = axes[4]
    # x.plot(time, outlet_H2O, label="H2O Outlet", color="tab:purple")
    # ax.plot(time, outlet_N2, label="N2 Outlet", color="tab:green")
    # ax.plot(time, outlet_O2, label="O2 Outlet", color="tab:orange")
    ax.plot(time, outlet_N2, label="N2 Outlet", color="tab:purple", linestyle="None", marker="o", markersize=2)
    # ax.plot(
    #     time_5,
    #     RH,
    #     color="black",
    #     linestyle="--",
    #     alpha=0.7,
    # )
    ax.set_title("N2 Gas Phase")
    ax.set_ylabel("Relative Humidity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. H2O adsorbed
    ax = axes[5]
    ax.plot(time, outlet_H2O, label="H2O Outlet", color="tab:blue")
    # ax.plot(
    #     time,
    #     qH2O,
        
    #     color="black",
    #     linestyle="--",
    #     alpha=0.7,
    # )
    ax.set_title("H2O Mole Fraction")
    ax.set_ylabel("H2O Mole Fraction")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Remove gridlines
    for ax in axes:
        ax.grid(False)

        # Add dashed vertical lines for stage transitions
    # Define stage durations (must match your run_cycle stages)
    stage_durations = [
        bed_properties["desorption_time"],
        bed_properties["cooling_time"],
        bed_properties["pressurisation_time"],
        bed_properties["adsorption_time"],
    ]
    stage_start_times = [0]
    for dur in stage_durations[:-1]:
        stage_start_times.append(stage_start_times[-1] + dur)

    for ax in axes:
        for t in stage_start_times:
            ax.axvline(x=t, color="#cccccc", linestyle="--", alpha=0.2)

    for ax in axes:
        ax.set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show()
