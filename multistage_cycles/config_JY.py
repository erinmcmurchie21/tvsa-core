import numpy as np
import matplotlib.pyplot as plt
from additional_functions_multistage import (
    create_non_uniform_grid,
    relative_humidity_to_mole_fraction,
    mole_fraction_to_relative_humidity
)


def create_fixed_properties():
    """
    Define fixed bed properties, grid configuration, and initial conditions.

    Returns:
        tuple: (bed_properties, column_grid, initial_conditions, rtol, atol_array)
    """
    # Column dimensions and physical properties
    bed_properties = {
        # Geometry
        "bed_length": 0.01,  # Bed length [m]
        "inner_bed_radius": 0.05,  # Inner radius [m]
        "outer_bed_radius": 0.051,  # Outer radius [m]
        "column_area": 0.05**2 * np.pi,  # Cross-sectional area [m²]
        # Porosity and density
        "bed_voidage": 0.4,  # Bed voidage [-]
        "particle_voidage": 0.23,  # Particle voidage [-]
        "total_voidage": 0.54,  # Total voidage [-]
        "bed_density": 880 * (1 - 0.4),  # Bed density [kg/m³]
        "sorbent_density": 880,  # Sorbent density [kg/m³]
        "wall_density": 2700,  # Wall density [kg/m³]
        # Transport properties
        "tortuosity": 3,  # Tortuosity factor [-]
        "molecular_diffusivity": 0.605e-5,  # Molecular diffusivity [m²/s]
        "particle_diameter": 0.00052,  # Particle diameter [m]
        "K_z": 0.1,  # Axial dispersion coefficient [m²/s]
        "K_wall": 205,  # Wall thermal conductivity [W/m·K]
        "mu": 1.78e-5,  # Gas viscosity [Pa·s]
        # Mass transfer coefficients
        "mass_transfer_1": 0.003,  # CO2 mass transfer coeff [s⁻¹]
        "mass_transfer_2": 0.0086,  # H2O mass transfer coeff [s⁻¹]
        "mass_transfer_3": 0.0,  # N2 mass transfer coeff [s⁻¹]
        # Heat transfer properties
        "h_bed": 14,  # Bed heat transfer coeff [W/m²·K]
        "h_wall": 26,  # Wall heat transfer coeff [W/m²·K]
        "sorbent_heat_capacity": 1580,  # Solid heat capacity [J/kg·K]
        "wall_heat_capacity": 4e6 / 1590,  # Wall heat capacity [J/kg·K]
        "heat_capacity_1": 42.46,  # CO2 adsorbed phase Cp [J/mol·K]
        "heat_capacity_2": 73.1,  # H2O adsorbed phase Cp [J/mol·K]
        "heat_capacity_3": 29.1,  # N2 adsorbed phase Cp [J/mol·K]
        # Molecular weights
        "MW_1": 44.01,  # CO2 [g/mol]
        "MW_2": 18.02,  # H2O [g/mol]
        "MW_3": 28.02,  # N2 [g/mol]
        "MW_4": 32.00,  # O2 [g/mol]
        # Thermodynamic properties
        "R": 8.314,  # Universal gas constant [J/mol·K]
        "k": 1.4,  # Heat capacity ratio [-]
        "ambient_temperature": 288.15,  # Ambient temperature [K]
        "ambient_pressure": 101325,  # Ambient pressure [Pa]
        # Optimisation parameters
        "desorption_temperature": 373.15,  # Desorption temperature [K]
        "vacuum_pressure": 20000,  # Vacuum pressure [Pa]
        "adsorption_time": 8000,  # Adsorption time [s]
        "blowdown_time": 60,  # Blowdown time [s]
        "heating_time": 2400,  # Heating time [s]
        "desorption_time": 20000,  # Desorption time [s]
        "cooling_time": 400,  # Cooling time [s]
        "pressurisation_time": 60,  # Pressurisation time [s]
        # Adsorption isotherms
        "isotherm_type_1": "WADST",  # CO2 isotherm type
        "isotherm_type_2": "GAB",  # H2O isotherm type
        "isotherm_type_3": "None",  # N2 isotherm type
        # Reference values for scaling (dimensionless variables)
        "P_ref": 101325,  # Reference pressure [Pa]
        "T_ref": 298.15,  # Reference temperature [K]
        "n_ref": 3000,  # Reference adsorbed amount [mol/m³]ß
        # Water properties
        "Cp_water": 4181.3,  # Heat capacity of water [J/kg.K]
        "Cp_steam": 2010,  # Heat capacity of steam [J/kg.K]
        "vaporization_energy": 2257e3,  # Latent heat of vaporization for water [J/kg]
        # Calculated properties
        "sorbent_mass": 0.01 * 0.05**2 * np.pi * 880,  # [kg]
        "sorbent_volume": 0.01 * 0.05**2 * np.pi * (1 - 0.4),  # [m³]
        "bed_volume": 0.01 * 0.05**2 * np.pi,  # [m³]
        # Efficiencies
        "compressor_efficiency": 0.75,  # Compressor efficiency
        "fan_efficiency": 0.5,  # Blower efficiency
        # Feed conditions
        "feed_velocity": 7.06e-2 * 2,  # Superficial feed velocity [m/s]
        "feed_temperature": 288.15,  # Feed temperature [K]
        "steam_velocity": 0,  # Superficial steam velocity [m/s]
        "steam_temperature": 368.15,  # Steam temperature [K]
        "humidity": 0.55,  # Relative humidity of feed [-]
        "feed_composition": {
            "y1": 0.0004,
            "y2": 0.0115,
            "y3": 0.9881,
        },
        "steam_composition": {
            "y1": 1e-6,
            "y2": 1 - 2e-6,
            "y3": 1e-6,
        },
        "stage_config": {
            
            "blowdown": {
                "left": "closed",
                "right": "pressure",
                "direction": "forwards",
            },
            "heating": {"left": "closed", "right": "pressure", "direction": "forwards"},
            "desorption": {
                "left": "closed",
                "right": "pressure",
                "direction": "forwards",
            },
            "cooling": {"left": "closed", "right": "closed", "direction": "forwards"},
            "pressurisation": {
                "left": "inlet_pressure",
                "right": "closed",
                "direction": "forwards",
            },
            "adsorption": {
                "left": "mass_flow",
                "right": "pressure",
                "direction": "forwards",
            },
        },
    }

    # Create spatial discretization grid
    column_grid = create_non_uniform_grid(bed_properties)

    # Initialize state variables
    num_cells = column_grid["num_cells"]

    # Initial conditions: ambient pressure, temperature, and composition
    y2_feed = relative_humidity_to_mole_fraction(
        0.55, bed_properties["ambient_pressure"], bed_properties["ambient_temperature"]
    )
    P_init = np.ones(num_cells) * 101325  # Pressure [Pa]
    T_init = np.ones(num_cells) * 288.15  # Gas temperature [K]
    Tw_init = np.ones(num_cells) * 288.15  # Wall temperature [K]
    y1_init = np.ones(num_cells) * 400e-6 / (1 - y2_feed)  # CO2 mole fraction
    y2_init = np.ones(num_cells) * y2_feed  # H2O mole fraction
    y3_init = np.ones(num_cells) * 0.95  # N2 mole fraction

    # Calculate initial adsorbed amounts from equilibrium isotherms
    n1_init = adsorption_isotherm_1(
        P_init,
        T_init,
        y1_init,
        y2_init,
        y3_init,
        400e-6,
        y2_init[0],
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
        n1_init,
        n2_init,
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

    bed_density = bed_properties["bed_density"]
    ε = bed_properties["bed_voidage"]
    q1_init = n1_init / (bed_density / (1 - ε))  # mol/m³
    print(
        f"Initial adsorbed CO2 loading (mid-bed): {q1_init[num_cells // 2]:.2f} mol/m³"
    )

    # Solver tolerance settings
    rtol = 1e-5  # Relative tolerance

    # Absolute tolerances for different variable types
    atol_P = 1e-4 * np.ones(len(P_init))  # Pressure
    atol_T = 1e-4 * np.ones(len(T_init))  # Temperature
    atol_Tw = 1e-4 * np.ones(len(Tw_init))  # Wall temperature
    atol_y1 = 1e-8 * np.ones(len(y1_init))  # CO2 mole fraction
    atol_y2 = 1e-8 * np.ones(len(y2_init))  # H2O mole fraction
    atol_y3 = 1e-8 * np.ones(len(y3_init))  # N2 mole fraction
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

    if isotherm_type_1 == "Toth":
        T_0 = 298.15  # Reference temperature (K)
        n_s = 4.86 * np.exp(0 * (1 - temperature / T_0))  # mol/kg
        b = 28.5 * np.exp(-1 * (-117798) / (R * T_0) * (1 - T_0 / temperature))  # kPa⁻¹
        t = 0.209 - 0.523 * (1 - T_0 / temperature)

        load_kg = (
            n_s * b * pressure * y1 / (1 + (b * pressure * y1) ** t) ** (1 / t)
        )  # mol/kg
        load_m3 = load_kg * bed_density / (1 - ε)  # mol/m³
        ΔH = -70000  # J/mol

    elif isotherm_type_1 == "ModifiedToth":
        T_0 = 298.15  # Reference temperature (K)
        n_s = 4.86 * np.exp(0 * (1 - temperature / T_0))  # mol/kg
        b = 2.85e-21 * np.exp((117798) / (R * temperature))  # Pa⁻¹
        t = 0.209 + 0.523 * (1 - T_0 / temperature)

        load_kg = (
            n_s * b * pressure * y1 / (1 + (b * pressure * y1) ** t) ** (1 / t)
        )  # mol/kg
        load_m3 = load_kg * bed_density / (1 - ε)  # mol/m³
        ΔH = -70000  # J/mol

    elif isotherm_type_1 == "Stampi-Bombelli":
        T_0 = 298.15
        n2_mass = adsorption_isotherm_2(
            pressure, temperature, y2, bed_properties, isotherm_type="GAB"
        )[0] / (bed_density / (1 - ε))  # mol/kg
        gamma = -0.137
        beta = 5.612  # Reference temperature (K)
        n_s = (
            4.86 * np.exp(0 * (1 - temperature / T_0)) * (1 / (1 - gamma * n2_mass))
        )  # mol/kg
        b = (
            28.5
            * np.exp(-1 * (-117798) / (R * T_0) * (1 - T_0 / temperature))
            * (1 + beta * n2_mass)
        )  # kPa⁻¹
        t = 0.209 - 0.523 * (1 - T_0 / temperature)

        load_kg = (
            n_s * b * pressure * y1 / (1 + (b * pressure * y1) ** t) ** (1 / t)
        )  # mol/kg
        load_m3 = load_kg * bed_density / (1 - ε)  # mol/m³
        ΔH = -70000  # J/mol

    elif isotherm_type_1 == "WADST":
        T_0 = 298.15  # Reference temperature (K)
        n2_mass = adsorption_isotherm_2(
            pressure, temperature, y2, bed_properties, isotherm_type="GAB"
        )[0] / (bed_density / (1 - ε))  # mol/kg
        A = 1.532

        n_s_dry = 4.86 * np.exp(0 * (1 - temperature / T_0))  # mol/kg
        b_dry = 2.85e-21 * np.exp((117798) / (R * temperature))  # Pa⁻¹
        t_dry = 0.209 + 0.523 * (1 - T_0 / temperature)

        n_s_wet = 9.035 * np.exp(0 * (1 - temperature / T_0))  # mol/kg
        b_wet = 1.23e-18 * np.exp((203687) / (R * temperature))  # Pa⁻¹
        t_wet = 0.053 + 0.053 * (1 - T_0 / temperature)

        load_kg = (1 - np.exp(-A / n2_mass)) * n_s_dry * b_dry * pressure * y1 / (
            1 + (b_dry * pressure * y1) ** t_dry
        ) ** (1 / t_dry) + np.exp(-A / n2_mass) * n_s_wet * b_wet * pressure * y1 / (
            1 + (b_wet * pressure * y1) ** t_wet
        ) ** (1 / t_wet)  # mol/kg
        load_m3 = load_kg * bed_density / (1 - ε)  # mol/m³
        ΔH = -70000  # J/mol

    elif isotherm_type_1 == "Langmuir":
        c1 = y1 * pressure / (R * temperature)  # mol/m3
        c2 = y3 * pressure / (R * temperature)  # mol/m3
        b1 = 8.65e-7 * np.exp(-(-36641.21) / (R * temperature))  # m3/mol
        b2 = 2.5e-6 * np.exp(-(-1.58e4) / (R * temperature))  # m3/mol
        d1 = 2.63e-8 * np.exp(-(-3590.66) / (R * temperature))  # m3/mol
        d2 = 0

        load_kg = 3.09 * b1 * c1 / (1 + b1 * c1 + b2 * c2) + 2.54 * d1 * c1 / (
            1 + d1 * c1 + d2 * c2
        )  # mol/kg
        load_m3 = load_kg * bed_density / (1 - ε)  # mol/m³
        ΔH = -57047  # J/mol

    elif isotherm_type_1 == "Dual_site Langmuir":
        c1 = y1 * pressure / (R * temperature)  # mol/m3
        b1 = 3.17e-6 * np.exp(-(-28.63e3) / (R * temperature))  # m3/mol
        b2 = 3.21e-6 * np.exp(-(-20.37e3) / (R * temperature))  # m3/mol
        qs1 = 0.44  # mol/kg
        qs2 = 6.10

        load_kg = qs1 * b1 * c1 / (1 + b1 * c1) + qs2 * b2 * c1 / (
            1 + b2 * c1
        )  # mol/kg

        load_m3 = load_kg * bed_density / (1 - ε)  # mol/m³
        ΔH = R * (
            -3391.58 + 273.2725 * n1 * (1 - bed_properties["bed_voidage"]) / bed_density
        )  # J/mol

    elif isotherm_type_1 == "Mechanistic_Young":
        T_0 = 298.15
        n2_mass = adsorption_isotherm_2(
            pressure, temperature, y2, bed_properties, isotherm_type="GAB"
        )[0] / (bed_density / (1 - ε))  # mol/kg
        A = 1.535  # Reference temperature (K)
        DH_dry = 117798
        DH_wet = 130155
        DH = (1 - np.exp(-A / n2_mass)) * DH_dry + np.exp(-A / n2_mass) * DH_wet

        n_s = 4.86 * np.exp(0 * (1 - temperature / T_0))  # mol/kg
        b = 2.85e-21 * np.exp((DH) / (R * temperature))  # Pa⁻¹
        t = 0.209 + 0.523 * (1 - T_0 / temperature)

        f_blocked_max = 0.433
        k_block = 0.795
        phi_max = 1.0
        phi_dry = 1.0
        n = 1.425
        f_blocked = f_blocked_max * (1 - np.exp(-k_block * n2_mass) ** n)
        phi_av = phi_max - f_blocked
        f = (
            n_s * b * pressure * y1 / (1 + (b * pressure * y1) ** t) ** (1 / t)
        )  # mol/kg
        phi = phi_dry + (phi_av - phi_dry) * (np.exp(-A / n2_mass))
        load_kg = phi / phi_dry * f  # mol/kg

        load_m3 = load_kg * bed_density / (1 - ε)  # mol/m³

        ΔH = -50000  # J/mol
    return load_m3, ΔH


def adsorption_isotherm_2(
    pressure, temperature, y2, bed_properties, isotherm_type="GAB"
):
    """
    GAB isotherm for H₂O.

    Returns:
        equilibrium_loading (mol/m³ of solid)
        adsorption_heat_2 (J/mol)
    """
    R = 8.314  # J/mol·K
    bed_density = bed_properties["bed_density"]  # kg/m³
    ε = bed_properties["bed_voidage"]  # bed void fraction

    if isotherm_type == "GAB":
        q_m = 3.63  # mol/kg
        E_1 = 47110 - np.exp(0.023744 * temperature)
        E2_9 = 57706 - 47.814 * temperature
        E_10 = -44.38 * temperature + 57220
        c = np.exp((E_1 - E_10) / (R * temperature))
        k = np.exp((E2_9 - E_10) / (R * temperature))

        P_sat = 611.21 * np.exp(
            (18.678 - ((temperature - 273.15) / 234.5))
            * (temperature - 273.15)
            / (temperature - 16.01)
        )  # Tetens equation for water, Pa
        RH = y2 * pressure / (P_sat)  # dimensionless

        load_kg = q_m * k * c * RH / ((1 - k * RH) * (1 + (c - 1) * k * RH))  # mol/kg
        load_m3 = load_kg * bed_density / (1 - ε)  # mol/m³

    elif isotherm_type == "None":
        load_m3 = 0 * pressure

    ΔH = -46000  # J/mol

    return load_m3, ΔH

def adsorption_isotherm_3(
    pressure, temperature, y1, y2, y3, n1, n2, bed_properties, isotherm_type_3="None"
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

    if isotherm_type_3 == "None":
        load_m3 = 0 * pressure
        ΔH = 0

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

    time_ = np.loadtxt(
        "multistage_cycles/JY_profiles/qCO2.csv",
        delimiter=",",
        usecols=0,
    )  # First column
    qCO2 = np.loadtxt(
        "multistage_cycles/JY_profiles/qCO2.csv",
        delimiter=",",
        usecols=1,
    )  # Second column

    time_2 = np.loadtxt('multistage_cycles/JY_profiles/yCO2.csv', delimiter=',', usecols=0)  # First column
    yCO2 = np.loadtxt('multistage_cycles/JY_profiles/yCO2.csv', delimiter=',', usecols=1)  # Second column

    time_1 = np.loadtxt(
        "multistage_cycles/JY_profiles/qH2O.csv",
        delimiter=",",
        usecols=0,
    )  # First column
    qH2O = np.loadtxt(
        "multistage_cycles/JY_profiles/qH2O.csv",
        delimiter=",",
        usecols=1,
    )  # Second column

    time_4 = np.loadtxt(
        "multistage_cycles/JY_profiles/temperature.csv",
        delimiter=",",
        usecols=0,
    )  # First column
    temperature = (
        np.loadtxt(
            "multistage_cycles/JY_profiles/temperature.csv",
            delimiter=",",
            usecols=1,
        )
        + 273.15
    )  # Second column

    time_5 = np.loadtxt(
        "multistage_cycles/JY_profiles/RH.csv",
        delimiter=",",
        usecols=0,
    )  # First column
    RH = np.loadtxt(
        "multistage_cycles/JY_profiles/RH.csv",
        delimiter=",",
        usecols=1,
    )  # Second column

    time_6 = np.loadtxt(
        "multistage_cycles/JY_profiles/pressure_outlet.csv",
        delimiter=",",
        usecols=0,
    )  # First column
    pressure = np.loadtxt(
        "multistage_cycles/JY_profiles/pressure_outlet.csv",
        delimiter=",",
        usecols=1,
    ) * 100000# Second column

    time = profiles["time"]
    T_gas = np.array(profiles["temperature"])
    T_wall = np.array(profiles["wall_temperature"])
    P_inlet = np.array(profiles["pressure_inlet"])
    P_outlet = np.array(profiles["pressure_outlet"])
    outlet_CO2 = np.array(profiles["outlet_CO2"])
    outlet_H2O = np.array(profiles["outlet_H2O"])
    adsorbed_CO2 = np.array(profiles["adsorbed_CO2"])
    adsorbed_H2O = np.array(profiles["adsorbed_H2O"])
    # outlet_N2 = np.array(profiles["outlet_N2"])
    # outlet_O2 = np.array(profiles["outlet_O2"])
    # equilibrium_CO2 = np.array(profiles["equilibrium_CO2"])

    bed_density = bed_properties["bed_density"]
    bed_voidage = bed_properties["bed_voidage"]

    adsorbed_CO2 = np.array(adsorbed_CO2) / (bed_density / (1 - bed_voidage))
    adsorbed_H2O = np.array(adsorbed_H2O) / (bed_density / (1 - bed_voidage))
    relative_humidity = mole_fraction_to_relative_humidity(outlet_H2O, P_outlet, T_gas)

    figsize = (15, 8)
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.ravel()

    # 1. Temperature overlay (gas and wall)
    ax = axes[0]
    ax.plot(time, T_gas, label="Gas Temp (mid)", color="tab:blue")
    ax.plot(time, T_wall, label="Wall Temp (mid)", color="tab:orange")
    ax.plot(
        time_4,
        temperature,
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
    ax.plot(time, P_inlet, label="Inlet Pressure", color="tab:green")
    ax.plot(time, P_outlet, label="Outlet Pressure", color="tab:red")
    ax.plot(
        time_6,
        pressure,
        color="black",
        linestyle="--",
        alpha=0.7,
    )
    ax.set_title("Pressure Profiles")
    ax.set_ylabel("Pressure (Pa)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. CO2 gas phase (outlet and mid)
    ax = axes[2]
    ax.plot(time, outlet_CO2, label="CO2 Outlet", color="tab:purple")
    ax.plot(time_2, yCO2, color="black", linestyle="--", alpha=0.7)
    ax.set_title("CO2 Gas Phase")
    ax.set_ylabel("CO2 Mole Fraction")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. CO2 adsorbed
    ax = axes[3]
    ax.plot(time, adsorbed_CO2, label="CO2 Adsorbed (mid)", color="tab:blue")
    ax.plot(
         time_,
         qCO2,
         color="black",
         linestyle="--",
         alpha=0.7,
    )
    ax.set_title("CO2 Adsorbed")
    ax.set_ylabel("CO2 Loading (mol/kg)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Relative Humidity
    ax = axes[4]
    # x.plot(time, outlet_H2O, label="H2O Outlet", color="tab:purple")
    # ax.plot(time, outlet_N2, label="N2 Outlet", color="tab:green")
    # ax.plot(time, outlet_O2, label="O2 Outlet", color="tab:orange")
    ax.plot(time, relative_humidity, label="Relative Humidity", color="tab:purple")
    ax.plot(
        time_5,
        RH,
        color="black",
        linestyle="--",
        alpha=0.7,
    )
    ax.set_title("Relative Humidity")
    ax.set_ylabel("Relative Humidity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. H2O adsorbed
    ax = axes[5]
    ax.plot(time, adsorbed_H2O, label="H2O Adsorbed (mid)", color="tab:blue")
    ax.plot(
        time_1,
        qH2O,
        label="GAB (Stampi-Bombelli)",
        color="black",
        linestyle="--",
        alpha=0.7,
    )
    ax.set_title("H2O Adsorbed")
    ax.set_ylabel("H2O Loading (mol/kg)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Remove gridlines
    for ax in axes:
        ax.grid(False)

        # Add dashed vertical lines for stage transitions
    # Define stage durations (must match your run_cycle stages)
    stage_durations = [
        bed_properties["adsorption_time"],
        bed_properties["blowdown_time"],
        bed_properties["heating_time"],
        bed_properties["desorption_time"],
        bed_properties["pressurisation_time"],
        # Add cooling time if used
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
