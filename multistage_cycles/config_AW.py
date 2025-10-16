import numpy as np
from additional_functions_multistage import (
    create_non_uniform_grid,
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
        "bed_length": 0.0181,  # Bed length [m]
        "inner_bed_radius": 0.145,  # Inner radius [m]
        "outer_bed_radius": 0.162,  # Outer radius [m]
        "column_area": 0.145**2 * np.pi,  # Cross-sectional area [m²]
        # Porosity and density
        "bed_voidage": 0.37,  # Bed voidage [-]
        "particle_voidage": 0.9616,  # Particle voidage [-]
        "total_voidage": 0.726,  # Total voidage [-]
        "bed_density": 55.4,  # Bed density [kg/m³]
        "sorbent_density": 61,  # Sorbent density [kg/m³]
        "wall_density": 7800,  # Wall density [kg/m³]
        # Transport properties
        "tortuosity": 3,  # Tortuosity factor [-]
        "molecular_diffusivity": 1.605e-5,  # Molecular diffusivity [m²/s]
        "particle_diameter": 0.0075,  # Particle diameter [m]
        "K_z": 0.243,  # Axial dispersion coefficient [m²/s]
        "K_wall": 205,  # Wall thermal conductivity [W/m·K]
        "mu": 1.78e-5,  # Gas viscosity [Pa·s]
        # Mass transfer coefficients
        "mass_transfer_1": 0.0002,  # CO2 mass transfer coeff [s⁻¹]
        "mass_transfer_2": 0.002,  # H2O mass transfer coeff [s⁻¹]
        # Heat transfer properties
        "h_bed": 14,  # Bed heat transfer coeff [W/m²·K]
        "h_wall": 26,  # Wall heat transfer coeff [W/m²·K]
        "sorbent_heat_capacity": 2000,  # Solid heat capacity [J/kg·K]
        "wall_heat_capacity": 513,  # Wall heat capacity [J/kg·K]
        "heat_capacity_1": 42.46,  # CO2 adsorbed phase Cp [J/mol·K]
        "heat_capacity_2": 42.46,  # H2O adsorbed phase Cp [J/mol·K]
        # Molecular weights
        "MW_1": 44.01,  # CO2 [g/mol]
        "MW_2": 18.02,  # H2O [g/mol]
        "MW_3": 28.02,  # N2 [g/mol]
        "MW_4": 32.00,  # O2 [g/mol]
        # Thermodynamic properties
        "R": 8.314,  # Universal gas constant [J/mol·K]
        "k": 1.4,  # Heat capacity ratio [-]
        "ambient_temperature": 293.15,  # Ambient temperature [K]
        "ambient_pressure": 100000,  # Ambient pressure [Pa]
        # Optimisation parameters
        "desorption_temperature": 368.15,  # Desorption temperature [K]
        "vacuum_pressure": 5000,  # Vacuum pressure [Pa]
        "adsorption_time": 13772,  # Adsorption time [s]
        "blowdown_time": 30,  # Blowdown time [s]
        "heating_time": 704,  # Heating time [s]
        "desorption_time": 30000,  # Desorption time [s]
        "pressurisation_time": 50,  # Pressurisation time [s]
        # Adsorption isotherms
        "isotherm_type_1": "ModifiedToth",  # CO2 isotherm type
        "isotherm_type_2": "GAB",  # H2O isotherm type
        # Reference values for scaling (dimensionless variables)
        "P_ref": 101325,  # Reference pressure [Pa]
        "T_ref": 298.15,  # Reference temperature [K]
        "n_ref": 3000,  # Reference adsorbed amount [mol/m³]
        # Water properties
        "Cp_water": 4181.3,  # Heat capacity of water [J/kg.K]
        "Cp_steam": 2010,  # Heat capacity of steam [J/kg.K]
        "vaporization_energy": 2257e3,  # Latent heat of vaporization for water [J/kg]
        # Calculated properties
        "sorbent_mass": 0.0181 * 0.145**2 * np.pi * 55.4,  # [kg]
        "sorbent_volume": 0.0181 * 0.145**2 * np.pi * (1 - 0.092),  # [m³]
        "bed_volume": 0.0181 * 0.145**2 * np.pi,  # [m³]
        # Efficiencies
        "compressor_efficiency": 0.75,  # Compressor efficiency
        "fan_efficiency": 0.5,  # Fan efficiency
        # Feed conditions
        "feed_velocity": 0.1,  # Superficial feed velocity [m/s]
        "feed_temperature": 293.15,  # Feed temperature [K]
        "steam_velocity": 0.025,  # Superficial steam velocity [m/s]
        "steam_temperature": 368.15,  # Steam temperature [K]
        "feed_composition": {
            "y1": 0.0004 / (1 - 0.0115),
            "y2": 0.0115,
            "y3": 0.9881 / (1 - 0.0115),
        },
        "steam_composition": {
            "y1": 1e-6,
            "y2": 1 - 2e-6,
            "y3": 1e-6,
        },
        "stage_config": {
            "adsorption": {
                "left": "mass_flow",
                "right": "pressure",
                "direction": "forwards",
            },
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
            "steam_desorption": {
                "left": "mass_flow",
                "right": "pressure",
                "direction": "forwards",
            },
            "pressurisation": {
                "left": "pressure",
                "right": "closed",
                "direction": "forwards",
            },
            "cooling": {
                "left": "mass_flow",
                "right": "closed",
                "direction": "forwards",
            },
        },
    }

    # Create spatial discretization grid
    column_grid = create_non_uniform_grid(bed_properties)

    # Initialize state variables
    num_cells = column_grid["num_cells"]

    # Initial conditions: ambient pressure, temperature, and composition
    y2_feed = 0.0115
    P_init = np.ones(num_cells) * 100000  # Pressure [Pa]
    T_init = np.ones(num_cells) * 293.15  # Gas temperature [K]
    Tw_init = np.ones(num_cells) * 293.15  # Wall temperature [K]
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
        0.0115,
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
            F_init,
            E_init,  # Additional variables
        ]
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
        T_0 = 296  # Reference temperature (K)
        n_s = 2.38 * np.exp(0 * (1 - temperature / T_0))  # mol/kg
        b = 70.74 * np.exp(-1 * (-57047) / (R * T_0) * (1 - T_0 / temperature))  # kPa⁻¹
        t = 0.4148 - 1.606 * (1 - T_0 / temperature)
        pressure_kPa = pressure / 1000  # Convert pressure from Pa to kPa

        load_kg = (
            n_s * b * pressure_kPa * y1 / (1 + (b * pressure_kPa * y1) ** t) ** (1 / t)
        )  # mol/kg
        load_m3 = load_kg * bed_density / (1 - ε)  # mol/m³
        ΔH = -57047  # J/mol

    elif isotherm_type_1 == "ModifiedToth":
        T_0 = 296  # Reference temperature (K)
        n2_mass = adsorption_isotherm_2(
            pressure, temperature, y2, bed_properties, isotherm_type="GAB"
        )[0] / (bed_density / (1 - ε))  # mol/kg
        gamma = 0.0016  # kg/mol
        beta = 59.1  # kg/mol
        n_s = (
            2.38 * np.exp(0 * (1 - temperature / T_0)) * (1 / (1 - gamma * n2_mass))
        )  # mol/kg
        b = (
            70.74
            * np.exp((-57047) / (R * T_0) * (T_0 / temperature - 1))
            * (1 + beta * n2_mass)
        )  # kPa⁻¹
        t = 0.4148 - 1.606 * (1 - T_0 / temperature)

        pressure_kPa = pressure / 1000  # Convert pressure from Pa to kPa

        load_kg = (
            n_s * b * pressure_kPa * y1 / (1 + (b * pressure_kPa * y1) ** t) ** (1 / t)
        )  # mol/kg
        load_m3 = load_kg * bed_density / (1 - ε)  # mol/m³
        ΔH = -57047  # J/mol

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
    bed_density = bed_properties["bed_density"]  # kg/m³
    ε = bed_properties["bed_voidage"]  # bed void fraction

    if isotherm_type == "GAB":
        K_ads = 0.5751  # -
        c_m = 36.48  # mol/kg
        c_G = 0.1489  # -

        P_sat = 611.21 * np.exp(
            (18.678 - ((temperature - 273.15) / 234.5))
            * (temperature - 273.15)
            / (temperature - 16.01)
        )  # Tetens equation for water, Pa
        RH = y2 * pressure / (P_sat)  # dimensionless

        load_kg = (
            c_m * c_G * K_ads * RH / ((1 - K_ads * RH) * (1 + (c_G - 1) * K_ads * RH))
        )  # mol/kg
        load_m3 = load_kg * bed_density / (1 - ε)  # mol/m³

    elif isotherm_type == "None":
        load_m3 = 0 * pressure

    ΔH = -49000  # J/mol

    return load_m3, ΔH
