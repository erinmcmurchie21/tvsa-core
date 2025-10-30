
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import additional_functions as func

import config_base as base

@dataclass
class CO2BinarySips(base.IsothermModel):
    """Binary Sips isotherm for CO2 on Zeolite 13X"""
    
    # CO2 parameters
    n_inf_base: float = 7.268  # mol/kg
    X_CO2: float = -0.61684  # [-]
    Qb_CO2: float = 28389.0  # J/mol
    b_CO2_pre: float = 1.129e-4  # bar⁻¹
    alpha_CO2: float = 0.72378  # [-]
    c_CO2_base: float = 0.42456  # [-]
    
    # N2 parameters (for competitive adsorption)
    Qb_N2: float = 18474.0  # J/mol
    b_N2_pre: float = 5.847e-5  # bar⁻¹
    alpha_N2: float = 0.0  # [-]
    c_N2_base: float = 0.98624  # [-]
    
    # Reference conditions
    T_ref: float = 298.15  # K
    heat_of_adsorption: float = -37000.0  # J/mol
    
    def loading(
        self,
        pressure: np.ndarray,
        temperature: np.ndarray,
        y_CO2: np.ndarray,
        y_H2O: np.ndarray,
        y_N2: np.ndarray,
        adsorbent: base.AdsorbentProperties
    ) -> Tuple[np.ndarray, float]:
        """Calculate CO2 loading using Binary Sips model"""
        R = 8.314  # J/(mol·K)
        
        # Temperature-dependent parameters
        T_ratio = temperature / self.T_ref
        
        # Maximum loading
        n_inf = self.n_inf_base * np.exp(self.X_CO2 * (T_ratio - 1))
        
        # CO2 parameters
        b_CO2 = self.b_CO2_pre * np.exp(self.Qb_CO2 / (R * temperature))
        p_CO2 = pressure * y_CO2 / 1e5  # Convert Pa to bar
        c_CO2 = self.c_CO2_base + self.alpha_CO2 * (T_ratio - 1)
        
        # N2 parameters
        b_N2 = self.b_N2_pre * np.exp(self.Qb_N2 / (R * temperature))
        p_N2 = pressure * y_N2 / 1e5  # Convert Pa to bar
        c_N2 = self.c_N2_base + self.alpha_N2 * (T_ratio - 1)
        
        # Binary Sips isotherm equation
        numerator = (b_CO2 * p_CO2) ** c_CO2
        denominator = 1 + (b_CO2 * p_CO2) ** c_CO2 + (b_N2 * p_N2) ** c_N2
        
        load_kg = n_inf * numerator / denominator  # mol/kg
        load_m3 = load_kg * adsorbent.bed_density  # mol/m³
        
        return load_m3, self.heat_of_adsorption


@dataclass
class N2BinarySips(base.IsothermModel):
    """Binary Sips isotherm for N2 on Zeolite 13X"""
    
    # N2 parameters
    n_inf_base: float = 4.051  # mol/kg
    X_N2: float = 0.0  # [-]
    
    # CO2 parameters (for competitive adsorption)
    Qb_CO2: float = 28389.0  # J/mol
    b_CO2_pre: float = 1.129e-4  # bar⁻¹
    alpha_CO2: float = 0.72378  # [-]
    c_CO2_base: float = 0.42456  # [-]
    
    # N2 parameters
    Qb_N2: float = 18474.0  # J/mol
    b_N2_pre: float = 5.847e-5  # bar⁻¹
    alpha_N2: float = 0.0  # [-]
    c_N2_base: float = 0.98624  # [-]
    
    # Reference conditions
    T_ref: float = 298.15  # K
    heat_of_adsorption: float = -18500.0  # J/mol
    
    def loading(
        self,
        pressure: np.ndarray,
        temperature: np.ndarray,
        y_CO2: np.ndarray,
        y_H2O: np.ndarray,
        y_N2: np.ndarray,
        adsorbent: base.AdsorbentProperties
    ) -> Tuple[np.ndarray, float]:
        """Calculate N2 loading using Binary Sips model"""
        R = 8.314  # J/(mol·K)
        
        # Temperature-dependent parameters
        T_ratio = temperature / self.T_ref
        
        # Maximum loading
        n_inf = self.n_inf_base * np.exp(self.X_N2 * (T_ratio - 1))
        
        # CO2 parameters (for competitive adsorption)
        b_CO2 = self.b_CO2_pre * np.exp(self.Qb_CO2 / (R * temperature))
        p_CO2 = pressure * y_CO2 / 1e5  # Convert Pa to bar
        c_CO2 = self.c_CO2_base + self.alpha_CO2 * (T_ratio - 1)
        
        # N2 parameters
        b_N2 = self.b_N2_pre * np.exp(self.Qb_N2 / (R * temperature))
        p_N2 = pressure * y_N2 / 1e5  # Convert Pa to bar
        c_N2 = self.c_N2_base + self.alpha_N2 * (T_ratio - 1)
        
        # Binary Sips isotherm equation
        numerator = (b_N2 * p_N2) ** c_N2
        denominator = 1 + (b_CO2 * p_CO2) ** c_CO2 + (b_N2 * p_N2) ** c_N2
        
        load_kg = n_inf * numerator / denominator  # mol/kg
        load_m3 = load_kg * adsorbent.bed_density  # mol/m³
        
        return load_m3, self.heat_of_adsorption

def create_configuration() -> Tuple[base.AdsorptionColumnConfig, np.ndarray, float, np.ndarray]:
    """
    Create the complete LJ_2015 column configuration.
    
    Returns:
        (config, initial_conditions, rtol, atol_array)
    """
    
    # -------------------------------------------------------------------------
    # 1. GEOMETRY
    # -------------------------------------------------------------------------
    geometry = base.ColumnGeometry(
        bed_length=1.2,  # m
        inner_bed_radius=0.015,  # m
        outer_bed_radius=0.016,  # m
        wall_density=7800.0,  # kg/m³
        wall_heat_capacity=513.0,  # J/(kg·K)
        wall_heat_capacity_volumetric=4e6  # J/(K·m³)
    )
    
    # -------------------------------------------------------------------------
    # 2. ADSORBENT PROPERTIES WITH LJ_2015 ISOTHERMS
    # -------------------------------------------------------------------------
    adsorbent = base.AdsorbentProperties(
        name="Zeolite 13X",
        particle_diameter=0.002,  # m
        sorbent_density=1085.0,  # kg/m³
        bed_density=708.0,  # kg/m³
        material_density=2359.0,  # kg/m³
        bed_voidage=0.3475,
        particle_voidage=0.5401,
        tortuosity=3.0,
        sorbent_heat_capacity=920.0,  # J/(kg·K)
        heat_capacity_CO2=42.46,  # J/(mol·K)
        heat_capacity_H2O=73.1,  # J/(mol·K)
        heat_capacity_N2=29.1,  # J/(mol·K)
        # Assign LJ_2015 specific isotherms
        CO2_isotherm=CO2BinarySips(),
        H2O_isotherm=base.NoAdsorption(),
        N2_isotherm=N2BinarySips()
    )
    
    # -------------------------------------------------------------------------
    # 3. TRANSPORT PROPERTIES
    # -------------------------------------------------------------------------
    transport = base.TransportProperties(
        molecular_diffusivity=1.605e-5,
        K_z=0.09,
        K_wall=50.0,
        mu=1.78e-5,
        mass_transfer_CO2=0.15,
        mass_transfer_H2O=0.5,
        mass_transfer_N2=0.0,
        h_bed=20.0,
        h_wall=100.0
    )
    
    # -------------------------------------------------------------------------
    # 4. FLUID PROPERTIES (use defaults)
    # -------------------------------------------------------------------------
    fluid = base.FluidProperties()
    
    # -------------------------------------------------------------------------
    # 5. FEED CONDITIONS
    # -------------------------------------------------------------------------
    feed = base.FeedConditions(
        temperature=300.0,  # K
        pressure=102000.0,  # Pa (1.3 bar)
        flow_rate=3.5e-4,  # m³/s
        velocity=0.50 * 0.3475,  # m/s
        composition={'y1': 0.12, 'y2': 0.0, 'y3': 0.88}
    )
    
    # -------------------------------------------------------------------------
    # 6. STEAM CONDITIONS
    # -------------------------------------------------------------------------
    steam = base.SteamConditions(
        temperature=298.0,
        velocity=None
    )
    
    # -------------------------------------------------------------------------
    # 7. STAGE CONFIGURATION
    # -------------------------------------------------------------------------
    stages = base.StageConfiguration(
        desorption=base.StageStep(
            left='closed',
            right='pressure',
            direction='forwards',
            duration=2100
        ),
        cooling=base.StageStep(
            left='closed',
            right='closed',
            direction='forwards',
            duration=1450
        ),
        pressurisation=base.StageStep(
            left='inlet_pressure',
            right='closed',
            direction='forwards',
            duration=50
        ),
        adsorption=base.StageStep(
            left='mass_flow',
            right='pressure',
            direction='forwards',
            duration=1200
        ),
    )
    
    # -------------------------------------------------------------------------
    # 8. OPERATION PARAMETERS
    # -------------------------------------------------------------------------
    operation = base.OperationParameters(
        desorption_temperature=430.0,
        cooling_temperature=310.0,
        vacuum_pressure=101325.0,
        pressurisation_pressure=102000.0,
        ambient_temperature=310.0,
        ambient_pressure=101325.0,
        outside_temperature=310.0
    )
    
    # -------------------------------------------------------------------------
    # 9. REFERENCE VALUES (use defaults)
    # -------------------------------------------------------------------------
    reference = base.ReferenceValues()
    
    # -------------------------------------------------------------------------
    # 10. EFFICIENCY PARAMETERS (use defaults)
    # -------------------------------------------------------------------------
    efficiency = base.EfficiencyParameters()
    
    # -------------------------------------------------------------------------
    # 11. CREATE COMPLETE CONFIGURATION
    # -------------------------------------------------------------------------
    config = base.AdsorptionColumnConfig(
        column_id="LJ-2015",
        geometry=geometry,
        adsorbent=adsorbent,
        transport=transport,
        fluid=fluid,
        feed=feed,
        steam=steam,
        stages=stages,
        operation=operation,
        reference=reference,
        efficiency=efficiency
    )
    
    # -------------------------------------------------------------------------
    # 12. CREATE GRID AND ATTACH TO CONFIG
    # -------------------------------------------------------------------------
    config.grid = func.create_non_uniform_grid(geometry.bed_length)
    num_cells = config.grid['num_cells']
    
    # -------------------------------------------------------------------------
    # 13. INITIALIZE STATE VARIABLES
    # -------------------------------------------------------------------------
    y2_feed = 1e-6
    P_init = np.ones(num_cells) * 130000.0  # Pa
    T_init = np.ones(num_cells) * 310.0  # K
    Tw_init = np.ones(num_cells) * 310.0  # K
    y1_init = np.ones(num_cells) * 0.1  # CO2
    y2_init = np.ones(num_cells) * y2_feed  # H2O
    y3_init = np.ones(num_cells) * 0.88  # N2
    
    # Calculate initial adsorbed amounts using the configured isotherms
    n1_init = base.calculate_CO2_loading(
        config, P_init, T_init, y1_init, y2_init, y3_init
    )[0]
    
    n2_init = base.calculate_H2O_loading(
        config, P_init, T_init, y1_init, y2_init, y3_init
    )[0]
    
    n3_init = base.calculate_N2_loading(
        config, P_init, T_init, y1_init, y2_init, y3_init
    )[0]
    
    # -------------------------------------------------------------------------
    # 14. ADDITIONAL STATE VARIABLES
    # -------------------------------------------------------------------------
    F_init = np.zeros(8)
    E_init = np.zeros(7)
    
    # -------------------------------------------------------------------------
    # 15. COMBINE INITIAL CONDITIONS (SCALED)
    # -------------------------------------------------------------------------
    initial_conditions = np.concatenate([
        P_init / reference.P_ref,
        T_init / reference.T_ref,
        Tw_init / reference.T_ref,
        y1_init,
        y2_init,
        y3_init,
        n1_init / reference.n_ref,
        n2_init / reference.n_ref,
        n3_init / reference.n_ref,
        F_init,
        E_init
    ])
    
    # -------------------------------------------------------------------------
    # 16. CREATE SOLVER TOLERANCES
    # -------------------------------------------------------------------------
    rtol, atol_array = base.create_tolerances(num_cells)
    
    return config, initial_conditions, rtol, atol_array


def create_multi_plot(config: base.AdsorptionColumnConfig, profiles: dict):
    """
    Create a grid of subplots for LJ_2015 validation.
    
    Args:
        config: AdsorptionColumnConfig object
        profiles: Dictionary containing simulation results
    """
    import matplotlib.pyplot as plt
    
    # Load reference data
    author = "LJ_profiles_2015"
    
    try:
        time_temp = np.loadtxt(f"multistage_cycles/{author}/temperature.csv", 
                               delimiter=",", usecols=0)
        temp = np.loadtxt(f"multistage_cycles/{author}/temperature.csv", 
                         delimiter=",", usecols=1)
        
        time_yCO2 = np.loadtxt(f"multistage_cycles/{author}/yCO2.csv", 
                               delimiter=",", usecols=0)
        yCO2 = np.loadtxt(f"multistage_cycles/{author}/yCO2.csv", 
                         delimiter=",", usecols=1)
        
        time_p = np.loadtxt(f"multistage_cycles/{author}/pressure.csv", 
                           delimiter=",", usecols=0)
        pressure = np.loadtxt(f"multistage_cycles/{author}/pressure.csv", 
                             delimiter=",", usecols=1)
        has_reference = True
    except FileNotFoundError:
        print("Reference data files not found. Plotting simulation results only.")
        has_reference = False
    
    # Extract simulation results
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
    
    # Convert to mol/kg
    adsorbed_CO2_kg = adsorbed_CO2 / config.adsorbent.bed_density
    adsorbed_H2O_kg = adsorbed_H2O / config.adsorbent.bed_density
    
    # Create figure
    figsize = (15, 8)
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.ravel()
    
    # 1. Temperature profiles
    ax = axes[0]
    ax.plot(time, T_gas, label="Gas Temp (mid)", color="tab:blue", 
            linestyle="None", marker="o", markersize=2)
    ax.plot(time, T_wall, label="Wall Temp (mid)", color="tab:orange", 
            linestyle="None", marker="o", markersize=2)
    if has_reference:
        ax.plot(time_temp, temp, color="black", linestyle="--", alpha=0.7, label="Reference")
    ax.set_title("Temperature Profiles")
    ax.set_ylabel("Temperature (K)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Pressure profiles
    ax = axes[1]
    ax.plot(time, P_inlet, label="Inlet Pressure", color="tab:green", 
            linestyle="None", marker="o", markersize=2)
    ax.plot(time, P_outlet, label="Outlet Pressure", color="tab:red", 
            linestyle="None", marker="o", markersize=2)
    if has_reference:
        ax.plot(time_p, pressure, color="black", linestyle="--", alpha=0.7, label="Reference")
    ax.set_title("Pressure Profiles")
    ax.set_ylabel("Pressure (Pa)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. CO2 gas phase
    ax = axes[2]
    ax.plot(time, outlet_CO2, label="CO2 Outlet", color="tab:purple", 
            linestyle="None", marker="o", markersize=2)
    if has_reference:
        ax.plot(time_yCO2, yCO2, color="black", linestyle="--", alpha=0.7, label="Reference")
    ax.set_title("CO2 Gas Phase")
    ax.set_ylabel("CO2 Mole Fraction")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. CO2 adsorbed
    ax = axes[3]
    ax.plot(time, adsorbed_CO2_kg, label="CO2 Adsorbed (mid)", color="tab:blue", 
            linestyle="None", marker="o", markersize=2)
    ax.set_title("CO2 Adsorbed")
    ax.set_ylabel("CO2 Loading (mol/kg)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. N2 gas phase
    ax = axes[4]
    ax.plot(time, outlet_N2, label="N2 Outlet", color="tab:purple", 
            linestyle="None", marker="o", markersize=2)
    ax.set_title("N2 Gas Phase")
    ax.set_ylabel("N2 Mole Fraction")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. H2O mole fraction
    ax = axes[5]
    ax.plot(time, outlet_H2O, label="H2O Outlet", color="tab:blue")
    ax.set_title("H2O Mole Fraction")
    ax.set_ylabel("H2O Mole Fraction")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Remove gridlines
    for ax in axes:
        ax.grid(False)
    
    # Add stage transition lines
    stage_durations = [step.duration for step in 
                      [config.stages.desorption, config.stages.cooling, 
                       config.stages.pressurisation, config.stages.adsorption]]
    stage_start_times = [0]
    for dur in stage_durations[:-1]:
        stage_start_times.append(stage_start_times[-1] + dur)
    
    for ax in axes:
        for t in stage_start_times:
            ax.axvline(x=t, color="#cccccc", linestyle="--", alpha=0.2)
        ax.set_xlabel("Time (s)")
    
    plt.tight_layout()
    plt.show()