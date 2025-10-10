import numpy as np
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class StageKPIs:
    """KPIs for a single stage"""
    stage_name: str
    duration: float  # seconds
    
    # Mass flows (mol)
    CO2_out: float
    H2O_out: float
    N2_out: float
    O2_out: float
    total_out: float
    CO2_in: float
    
    # Energy
    heat_supplied: float  # J
    vacuum_work_done: float  # J
    fan_work_done: float  # J
    simulation_time: float  # seconds

@dataclass
class CycleKPIs:
    """KPIs for a complete cycle"""
    cycle_number: int
    total_duration: float  # seconds
    
    # Overall mass balance
    total_CO2_captured: float  # mol (from heating stage)
    total_CO2_fed: float  # mol (from adsorption stage)
    total_carrier_gas_used: float  # mol
    
    # Performance metrics
    purity_wet: float  # fraction (CO2 in product)
    purity_dry: float  # fraction (CO2 in product)
    recovery: float  # fraction (CO2 captured / CO2 fed)
    productivity: float  # mol CO2 / m³ sorbent / hour
    bed_size_factor: float  # kg sorbent / kg CO2 per day
    
    # Energy metrics
    total_heating_energy: float  # J
    total_vacuum_energy: float  # J
    total_fan_energy: float  # J
    specific_thermal_energy: float  # J / mol CO2
    specific_mechanical_energy: float  # J / mol CO2
    total_specific_energy: float  # MJ / kg CO2
    eq_work: float  # MJ / kg CO2

    CO2_production_rate: float  # kg CO2 / m³ sorbent / hour
    total_simulation_time: float  # seconds
    # Stage breakdown
    stage_kpis: Dict[str, StageKPIs]
    
    # Convergence
    cycle_error: float


def calculate_stage_kpis(stage_name: str, F_result: np.ndarray, time_array: np.ndarray,
                         E_result: np.ndarray, simulation_time: np.ndarray, bed_properties: dict) -> StageKPIs:
    """
    Calculate KPIs for a single stage.
    
    Parameters:
    -----------
    stage_name : str
        Name of the stage (e.g., 'adsorption', 'blowdown', 'heating', etc.)
    F_result : np.ndarray
        Flow results array from simulation [8 x time_steps]
        F[0]: CO2 in, F[1]: H2O in, F[2]: N2 in, F[3]: O2 in
        F[4]: CO2 out, F[5]: H2O out, F[6]: N2 out, F[7]: O2 out
    time_array : np.ndarray
        Time points
    E_result : np.ndarray
        Energy results array [3 x time_steps]
        E[0]: heat in, E[1]: heat out, E[2]: net energy
    bed_properties : dict
        Dictionary containing bed properties
    
    Returns:
    --------
    StageKPIs object
    """
    duration = time_array[-1] - time_array[0]
    
    # Calculate molar flows out (integrate over time)
    CO2_out = F_result[4, -1]
    H2O_out = F_result[5, -1]
    N2_out = F_result[6, -1]
    O2_out = F_result[7, -1]
    total_out = CO2_out + H2O_out + N2_out + O2_out
    
    # Calculate molar flows in
    CO2_in = F_result[0, -1]
    
    # Calculate energy (integrate over time)
    heat_supplied = E_result[3, -1]
    vacuum_work_done = E_result[4, -1]
    fan_work_done = E_result[5, -1]
    simulation_time = simulation_time

    return StageKPIs(
        stage_name=stage_name,
        duration=duration,
        CO2_out=CO2_out,
        H2O_out=H2O_out,
        N2_out=N2_out,
        O2_out=O2_out,
        total_out=total_out,
        CO2_in=CO2_in,
        heat_supplied=heat_supplied,
        fan_work_done=fan_work_done,
        vacuum_work_done=vacuum_work_done,
        simulation_time=simulation_time
    )


def calculate_cycle_kpis(cycle_number: int, stage_kpis_dict: Dict[str, StageKPIs],
                         bed_properties: dict, cycle_error: float) -> CycleKPIs:
    """
    Calculate overall cycle KPIs from individual stage KPIs.
    
    Parameters:
    -----------
    cycle_number : int
        Current cycle number
    stage_kpis_dict : Dict[str, StageKPIs]
        Dictionary of stage KPIs
    bed_properties : dict
        Dictionary containing bed properties
    cycle_error : float
        Convergence error for this cycle
    
    Returns:
    --------
    CycleKPIs object
    """
    # Total cycle duration
    total_duration = sum(stage.duration for stage in stage_kpis_dict.values())
    total_simulation_time = sum(stage.simulation_time for stage in stage_kpis_dict.values())
    MW_CO2 = bed_properties.get('MW_1', 44.01)  # g/mol

    # Product is from desorption stage
    desorption_stage = stage_kpis_dict.get('desorption', None)
    if desorption_stage:
        total_CO2_captured = desorption_stage.CO2_out
    kg_CO2_captured = total_CO2_captured * MW_CO2 / 1000

    # Carrier gas in desorption stage
    if desorption_stage:
        total_carrier_gas = desorption_stage.N2_out + desorption_stage.O2_out + desorption_stage.H2O_out
        dry_content_carrier_gas = desorption_stage.N2_out + desorption_stage.O2_out

    # Calculate CO2 fed 
    for stage_name in ['adsorption', 'pressurisation']:
        if stage_name in stage_kpis_dict:
            stage = stage_kpis_dict[stage_name]
            total_CO2_fed = stage.CO2_in
    
    # Performance metrics
    purity_wet = total_CO2_captured / (total_CO2_captured + total_carrier_gas)
    purity_dry = total_CO2_captured / (total_CO2_captured + dry_content_carrier_gas) 
    recovery = total_CO2_captured / total_CO2_fed
    
    # Productivity (mol CO2 / m³ sorbent / s)
    if total_duration > 0:
        productivity = total_CO2_captured / bed_properties["bed_volume"] / (total_duration)
        CO2_production_rate = kg_CO2_captured / (total_duration) * 3600  # kg/hr
    else:
        productivity = 0.0

    # Bed size factor (kg sorbent / kg CO2 per day)
    
    sorbent_mass = bed_properties.get('sorbent_mass', 1.0)  # kg
    if total_CO2_captured > 1e-10 and total_duration > 0:
        kg_CO2_per_day = total_CO2_captured * MW_CO2 / 1000 * (86400 / total_duration)
        bed_size_factor = sorbent_mass / kg_CO2_per_day
    else:
        bed_size_factor = 0.0
    
    # Energy metrics
    for stage_name in ['heating']:
        if stage_name in stage_kpis_dict:
            stage = stage_kpis_dict[stage_name]
            total_heating_energy = stage.heat_supplied  # J
            Q_thermal = stage.heat_supplied / total_CO2_captured  # J/mol
    for stage_name in ['adsorption']:
        if stage_name in stage_kpis_dict:
            stage = stage_kpis_dict[stage_name]
            total_fan_energy = stage.fan_work_done
            W_fan = stage.fan_work_done / total_CO2_captured  # J/mol
    for stage_name in ['cooling', 'blowdown', 'pressurisation', 'desorption']:
        if stage_name in stage_kpis_dict:
            stage = stage_kpis_dict[stage_name]
            total_vacuum_energy = stage.vacuum_work_done
            W_vacuum = stage.vacuum_work_done / total_CO2_captured  # J/mol
    
    # Specific energy (MJ / kg CO2 captured)

    specific_thermal_energy = total_heating_energy / 1e6 / kg_CO2_captured  # MJ/kg
    specific_mechanical_energy = (total_fan_energy + total_vacuum_energy) / 1e6 / kg_CO2_captured  # MJ/kg
    total_specific_energy = (specific_mechanical_energy + bed_properties["compressor_efficiency"]
                        * (1 - bed_properties["ambient_temperature"]/bed_properties['desorption_temperature']) 
                        * specific_thermal_energy)  # MJ/kgCO2
    
    # Equivalent work (J/mol)
    W = W_fan + W_vacuum
    W_eq = (W + bed_properties["compressor_efficiency"]
                        * (1 - bed_properties["ambient_temperature"]/bed_properties['desorption_temperature']) 
                        * Q_thermal)  # J/mol
    eq_work = W_eq / 1e3 / 44.01 # MJ/kg CO2

    return CycleKPIs(
        cycle_number=cycle_number,
        total_duration=total_duration,
        total_CO2_captured=total_CO2_captured,
        total_CO2_fed=total_CO2_fed,
        total_carrier_gas_used=total_carrier_gas,
        purity_wet=purity_wet,
        purity_dry=purity_dry,
        recovery=recovery,
        productivity=productivity,
        bed_size_factor=bed_size_factor,
        total_heating_energy=total_heating_energy,
        total_fan_energy=total_fan_energy,
        total_vacuum_energy=total_vacuum_energy,
        specific_thermal_energy=specific_thermal_energy,
        specific_mechanical_energy=specific_mechanical_energy,
        total_specific_energy=total_specific_energy,
        eq_work = W_eq,
        CO2_production_rate=CO2_production_rate,
        stage_kpis=stage_kpis_dict,
        cycle_error=cycle_error,
        total_simulation_time=total_simulation_time
    )


def print_stage_kpis(stage_kpis: StageKPIs):
    """Print formatted stage KPIs"""
    #print("    Stage Summary:")
    #print(f"    Total thermal energy supplied: {stage_kpis.heat_supplied/1e6:.2e} MJ")
    #print(f"    Total fan work done: {stage_kpis.fan_work_done/1e6:.2e} MJ")
    #print(f"    Total vacuum work done: {stage_kpis.vacuum_work_done/1e6:.2e} MJ")


def print_cycle_kpis(cycle_kpis: CycleKPIs):
    """Print formatted cycle KPIs"""
    print(f"\n{'#'*60}")
    print(f"CYCLE {cycle_kpis.cycle_number} SUMMARY")
    print(f"{'#'*60}")
    print(f"\nTotal Cycle Duration: {cycle_kpis.total_duration:.2f} s ({cycle_kpis.total_duration/60:.2f} min)")
    print(f"Total Simulation Time: {cycle_kpis.total_simulation_time:.2f} s ({cycle_kpis.total_simulation_time/60:.2f} min)")
    
    print("\nPERFORMANCE METRICS")
    print(f"{'─'*60}")
    print(f"Purity (wet):       {cycle_kpis.purity_wet*100:6.2f}%")
    print(f"Purity (dry):       {cycle_kpis.purity_dry*100:6.2f}%")
    print(f"Recovery:            {cycle_kpis.recovery*100:6.2f}%")
    print(f"Productivity:        {cycle_kpis.productivity:10.4f} molCO2/m³/s")
    print(f"CO2 Production Rate: {cycle_kpis.CO2_production_rate:10.4f} kgCO2/hr")
    
    print("\nENERGY METRICS")
    print(f"{'─'*60}")
    print(f"Specific Thermal Energy:   {cycle_kpis.specific_thermal_energy:.2e} MJ/kgCO2")
    print(f"Specific Mechanical Energy: {cycle_kpis.specific_mechanical_energy:.2e} MJ/kgCO2")
    print(f"Total Specific Energy:      {cycle_kpis.total_specific_energy:.2e} MJ/kgCO2")
    #print(f"Equivalent Work:            {cycle_kpis.eq_work:.2e} MJ/kg CO2")
    
    print("\nCONVERGENCE")
    print(f"{'─'*60}")
    print(f"Cycle Error:         {cycle_kpis.cycle_error:.2e}")


def export_kpis_to_dict(cycle_kpis: CycleKPIs) -> dict:
    """Export cycle KPIs to a dictionary for easy data analysis"""
    kpis_dict = {
        'cycle_number': cycle_kpis.cycle_number,
        'total_duration_s': cycle_kpis.total_duration,
        'CO2_captured_mol': cycle_kpis.total_CO2_captured,
        'CO2_fed_mol': cycle_kpis.total_CO2_fed,
        'purity': cycle_kpis.purity,
        'recovery': cycle_kpis.recovery,
        'productivity_mol_m3_hr': cycle_kpis.productivity,
        'bed_size_factor': cycle_kpis.bed_size_factor,
        'specific_energy_MJ_kg': cycle_kpis.specific_energy,
        'cycle_error': cycle_kpis.cycle_error
    }
    
    # Add stage-specific data
    for stage_name, stage in cycle_kpis.stage_kpis.items():
        kpis_dict[f'{stage_name}_duration_s'] = stage.duration
        kpis_dict[f'{stage_name}_CO2_out_mol'] = stage.CO2_out
        kpis_dict[f'{stage_name}_purity'] = stage.y_CO2_out
        kpis_dict[f'{stage_name}_energy_MJ'] = stage.net_energy / 1e6
    
    return kpis_dict