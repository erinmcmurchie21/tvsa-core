"""
Corrected configuration file combining the best of both approaches:
- Type-safe dataclasses from config_template.py
- Working functions from config_LJ_2015.py
- Factory function to create complete configuration
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from pyparsing import ABC, abstractmethod

# ============================================================================
# BASE ISOTHERM CLASS
# ============================================================================

class IsothermModel(ABC):
    """
    Abstract base class for all isotherm models.
    Each configuration (LJ_2015, etc.) will define their own subclasses.
    """
    
    @abstractmethod
    def loading(
        self,
        pressure: np.ndarray,
        temperature: np.ndarray,
        y_CO2: np.ndarray,
        y_H2O: np.ndarray,
        y_N2: np.ndarray,
        adsorbent: 'AdsorbentProperties'
    ) -> Tuple[np.ndarray, float]:
        """
        Calculate equilibrium loading.
        
        Returns:
            Tuple of (loading [mol/m³], heat_of_adsorption [J/mol])
        """
        pass


@dataclass
class NoAdsorption(IsothermModel):
    """Placeholder for non-adsorbing components"""
    heat_of_adsorption: float = 0.0
    
    def loading(self, pressure, temperature, y_CO2, y_H2O, y_N2, adsorbent):
        return 0.0 * pressure, self.heat_of_adsorption


# ============================================================================
# DATACLASS DEFINITIONS
# ============================================================================

@dataclass
class ColumnGeometry:
    """Physical dimensions of the adsorption column"""
    
    bed_length: float  # meters
    inner_bed_radius: float  # meters
    outer_bed_radius: float  # meters
    wall_density: float  # kg/m³
    wall_heat_capacity: float # J/kg·K
    wall_heat_capacity_volumetric: float  # J/(K·m³)
    
    @property
    def column_area(self) -> float:
        """Calculate column cross-sectional area in m²"""
        return np.pi * self.inner_bed_radius**2
    
    @property
    def bed_volume(self) -> float:
        """Calculate bed volume in m³"""
        return self.column_area * self.bed_length


@dataclass
class AdsorbentProperties:
    """Properties of the adsorbent material"""
    
    name: str
    particle_diameter: float  # meters
    sorbent_density: float  # kg/m³ (particle density)
    bed_density: float  # kg/m³
    material_density: float  # kg/m³ (13X material density)
    bed_voidage: float  # void fraction (0-1)
    particle_voidage: float  # void fraction within particles (0-1)
    tortuosity: float # dimensionless
    
    # Heat capacities
    sorbent_heat_capacity: float  # J/kg·K
    heat_capacity_CO2: float  # J/mol·K (adsorbed phase)
    heat_capacity_H2O: float  # J/mol·K (adsorbed phase)
    heat_capacity_N2: float # J/mol·K (adsorbed phase)

    # Isotherm models (set by specific configuration)
    CO2_isotherm: Optional[IsothermModel] = None
    H2O_isotherm: Optional[IsothermModel] = None
    N2_isotherm: Optional[IsothermModel] = None

    @property
    def total_voidage(self) -> float:
        """Calculate total voidage of the bed"""
        return self.bed_voidage + (1 - self.bed_voidage) * self.particle_voidage
    
    @property
    def sorbent_mass(self) -> float:
        """Calculate total sorbent mass (requires geometry)"""
        # This will be set after initialization
        return self.bed_density * ColumnGeometry.bed_volume
    


@dataclass
class TransportProperties:
    """Transport properties for the gas mixture"""
    
    molecular_diffusivity: float = 1.605e-5  # m²/s
    K_z: float  # axial dispersion coefficient, m²/s
    K_wall: float # wall thermal conductivity, W/m·K
    mu: float = 1.78e-5  # gas viscosity, Pa·s
    
    # Mass transfer coefficients
    mass_transfer_CO2: float  # s⁻¹
    mass_transfer_H2O: float  # s⁻¹
    mass_transfer_N2: float  # s⁻¹
    
    # Heat transfer coefficients
    h_bed: float # bed heat transfer coefficient, W/m²·K
    h_wall: float # wall heat transfer coefficient, W/m²·K


@dataclass
class FluidProperties:
    """Fluid properties for the gas mixture"""
    
    MW_CO2: float = 44.01  # g/mol
    MW_H2O: float = 18.02  # g/mol
    MW_N2: float = 28.02  # g/mol
    MW_O2: float = 32.00  # g/mol
    
    R: float = 8.314  # J/(mol·K) - universal gas constant
    k: float = 1.4  # adiabatic index
    
    # Water properties
    Cp_water: float = 4181.3  # J/kg·K
    Cp_steam: float = 2010.0  # J/kg·K
    vaporization_energy: float = 2257e3  # J/kg


@dataclass
class FeedConditions:
    """Feed conditions for the adsorption column"""
    
    temperature: float  # K
    pressure: float  # Pa
    flow_rate: float  # m³/s
    velocity: float  # m/s (superficial)
    composition: Dict[str, float]  # {'y1': CO2, 'y2': H2O, 'y3': N2}


@dataclass
class SteamConditions:
    """Steam conditions for regeneration"""
    
    temperature: float = 298.0  # K
    velocity: Optional[float] = None  # m/s
    composition: Dict[str, float] = field(default_factory=lambda: {
        'y1': 1e-6,
        'y2': 1 - 2e-6,
        'y3': 1e-6
    })

@dataclass
class InitialConditions:
    """Initial conditions for the adsorption column"""
    P_init: np.ndarray  # Pa
    T_init: np.ndarray  # K
    Tw_init: np.ndarray  # K
    yCO2_init: np.ndarray  # mole fraction
    yH2O_init: np.ndarray  # mole fraction
    yN2_init: np.ndarray  # mole fraction
    nCO2_init: np.ndarray  # mol/kg
    nH2O_init: np.ndarray  # mol/kg
    nN2_init: np.ndarray  # mol/kg
    F_init : np.ndarray = np.zeros(8)  # mol/s
    E_init : np.ndarray = np.zeros(7)  # J/s

    @property
    def initial_state_vector(self) -> np.ndarray:
        """Combine all initial conditions into a single state vector"""
        return np.concatenate([
            self.P_init,
            self.T_init,
            self.Tw_init,
            self.yCO2_init,
            self.yH2O_init,
            self.yN2_init,
            self.nCO2_init,
            self.nH2O_init,
            self.nN2_init
        ])
    
@dataclass
class SimulationParameters:
    """Parameters for the simulation"""
    r_tol: float  # tolerance for numerical solver

    atol_P: np.ndarray  # absolute tolerance for pressure
    atol_T: np.ndarray  # absolute tolerance for temperature
    atol_Tw: np.ndarray  # absolute tolerance for wall temperature
    atol_yCO2: np.ndarray  # absolute tolerance for CO2 mole fraction
    atol_yH2O: np.ndarray  # absolute tolerance for H2O mole fraction
    atol_yN2: np.ndarray  # absolute tolerance for N2 mole fraction
    atol_nCO2: np.ndarray  # absolute tolerance for adsorbed CO2
    atol_nH2O: np.ndarray  # absolute tolerance for adsorbed H2O
    atol_nN2: np.ndarray  # absolute tolerance for adsorbed N2  
    atol_F: np.ndarray  # absolute tolerance for molar flow rates
    atol_E: np.ndarray  # absolute tolerance for energy flow rates

    @property
    def atol_vector(self) -> np.ndarray:
        """Combine all absolute tolerances into a single vector"""
        return np.concatenate([
            self.atol_P,
            self.atol_T,
            self.atol_Tw,
            self.atol_yCO2,
            self.atol_yH2O,
            self.atol_yN2,
            self.atol_nCO2,
            self.atol_nH2O,
            self.atol_nN2
        ])

@dataclass
class StageStep:
    """Single stage boundary configuration."""
    left: str
    right: str
    direction: str
    duration: float
    temperature: Optional[float] = None 

@dataclass
class StageConfiguration:
    """Container for all stages. Implements dict-like access by stage name."""
    desorption: StageStep
    cooling: StageStep
    pressurisation: StageStep
    adsorption: StageStep

    def __getitem__(self, key: str) -> Any:
        """Allow stage_config[stage_name] style access used elsewhere."""
        return getattr(self, key)

@dataclass
class OperationParameters:
    # Process parameters
    desorption_temperature: float # K
    cooling_temperature: float  # K
    low_pressure: float  # Pa
    high_pressure: float  # Pa
    
    ambient_temperature: float = 298.15  # K
    ambient_pressure: float = 101325.0  # Pa
    outside_temperature: float = 298.15  # K

@dataclass
class ReferenceValues:
    """Reference values for dimensionless scaling"""
    
    P_ref: float = 101325.0  # Pa
    T_ref: float = 298.15  # K
    n_ref: float = 3000.0  # mol/m³

@dataclass
class EfficiencyParameters:
    """Efficiency parameters for compressors and fans"""
    
    compressor_efficiency: float = 0.75  # [-]
    fan_efficiency: float = 0.5  # [-]


@dataclass
class AdsorptionColumnConfig:
    """
    Complete configuration for an adsorption column.
    All properties accessible via config.property_group.property
    """
    
    column_id: str
    geometry: ColumnGeometry
    adsorbent: AdsorbentProperties
    transport: TransportProperties
    fluid: FluidProperties
    feed: FeedConditions
    steam: SteamConditions
    stages: StageConfiguration
    operation: OperationParameters
    reference: ReferenceValues
    efficiency: EfficiencyParameters
    
    # Grid will be set after creation
    grid: Optional[Dict] = None

# ============================================================================
# ISOTHERM FUNCTIONS (CORRECTED)
# ============================================================================

def calculate_CO2_loading(
    config: AdsorptionColumnConfig,
    pressure: np.ndarray,
    temperature: np.ndarray,
    y_CO2: np.ndarray,
    y_H2O: np.ndarray = 0.0,
    y_N2: np.ndarray = 0.0,
) -> Tuple[np.ndarray, float]:
    """
    Calculate CO2 equilibrium loading using the configured isotherm model.
    
    Args:
        config: Complete column configuration
        pressure: Total pressure [Pa]
        temperature: Temperature [K]
        y_CO2: CO2 mole fraction
        y_H2O: H2O mole fraction
        y_N2: N2 mole fraction
    
    Returns:
        (loading [mol/m³], heat_of_adsorption [J/mol])
    """
    if config.adsorbent.CO2_isotherm is None:
        raise ValueError("CO2 isotherm not configured")
    
    return config.adsorbent.CO2_isotherm.loading(
        pressure, temperature, y_CO2, y_H2O, y_N2, config.adsorbent
    )


def calculate_H2O_loading(
    config: AdsorptionColumnConfig,
    pressure: np.ndarray,
    temperature: np.ndarray,
    y_CO2: np.ndarray = 0.0,
    y_H2O: np.ndarray = 0.0,
    y_N2: np.ndarray = 0.0,
) -> Tuple[np.ndarray, float]:
    """Calculate H2O equilibrium loading"""
    if config.adsorbent.H2O_isotherm is None:
        raise ValueError("H2O isotherm not configured")
    
    return config.adsorbent.H2O_isotherm.loading(
        pressure, temperature, y_CO2, y_H2O, y_N2, config.adsorbent
    )


def calculate_N2_loading(
    config: AdsorptionColumnConfig,
    pressure: np.ndarray,
    temperature: np.ndarray,
    y_CO2: np.ndarray = 0.0,
    y_H2O: np.ndarray = 0.0,
    y_N2: np.ndarray = 0.0,
) -> Tuple[np.ndarray, float]:
    """Calculate N2 equilibrium loading"""
    if config.adsorbent.N2_isotherm is None:
        raise ValueError("N2 isotherm not configured")
    
    return config.adsorbent.N2_isotherm.loading(
        pressure, temperature, y_CO2, y_H2O, y_N2, config.adsorbent
    )


def create_tolerances(num_cells: int) -> Tuple[float, np.ndarray]:
    """
    Create solver tolerance arrays.
    
    Args:
        num_cells: Number of spatial discretization cells
    
    Returns:
        (rtol, atol_array)
    """
    rtol = 1e-5
    
    atol_P = 1e-5 * np.ones(num_cells)
    atol_T = 1e-5 * np.ones(num_cells)
    atol_Tw = 1e-5 * np.ones(num_cells)
    atol_y1 = 1e-9 * np.ones(num_cells)
    atol_y2 = 1e-9 * np.ones(num_cells)
    atol_y3 = 1e-9 * np.ones(num_cells)
    atol_n1 = 1e-4 * np.ones(num_cells)
    atol_n2 = 1e-4 * np.ones(num_cells)
    atol_n3 = 1e-4 * np.ones(num_cells)
    atol_F = 1e-4 * np.ones(8)
    atol_E = 1e-4 * np.ones(7)
    
    atol_array = np.concatenate([
        atol_P, atol_T, atol_Tw,
        atol_y1, atol_y2, atol_y3,
        atol_n1, atol_n2, atol_n3,
        atol_F, atol_E
    ])
    
    return rtol, atol_array