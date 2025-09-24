""" Optimisation code for multistage adsorption cycles """

from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import multistage_cycles.additional_functions_multistage as funcs
import multistage_cycles.tvsa_adsorption_column_multistage as column
import multistage_cycles.run_adsorption_multistage as cycle
import time
import os
from scipy.optimize import differential_evolution

DecisionVector = namedtuple("DecisionVector", [
    "adsorption", # adsorption duration [s]
    "blowdown", # blowdown duration [s]
    "heating", # heating duration [s]
    "desorption", # desorption duration [s]
    "pressurisation", # pressurisation duration [s]
    "T_des", # desorption temperature [K]
    "P_vac" # vacuum pressure [Pa]
    ])

def run_cycle_with_params(durations, desorption_temp, vacuum_pressure, n_cycles):

    """
    Parameters
    ----------
    durations : dict
        {'adsorption': sec, 'blowdown': sec, 'heating': sec, 'pressurisation': sec}
    desorption_temperature : float
        Temperature in K for the heating/desorption stage.
    vacuum_pressure : float
        Target vacuum pressure in Pa used during blowdown/heating.
    n_cycles : int
        Number of cycles to run (use small number for optimisation iterations, e.g. 1 or 2).

    Returns
    -------
    dict or None
        {'net_CO2_mols', 'thermal_energy', 'vacuum_energy', 'total_energy', 'profiles'} or None on failure
    """
    
    bed_properties, column_grid, initial_conditions, rtol, atol_array = cycle.create_fixed_properties()

    bed_props= bed_properties.copy()
    bed_props["desorption_temperature"] = desorption_temp
    bed_props["vacuum_pressure"] = vacuum_pressure


    # Initial conditions
    current_initial_conditions = initial_conditions.copy()

    # storage
    cycle_profiles = {'time': [], 'temperature': [], 'pressure_inlet': [], 'pressure_outlet': [],
                      'mass_CO2_out': [], 'mass_carrier_gas_out': [], 'mass_CO2_in': [], 'mass_H2O_out': [],
                      'adsorbed_CO2': [], 'adsorbed_H2O': [], 'wall_temperature': [],
                      'thermal_energy_input': [], 'vacuum_energy_input': [], 'fan_energy_input': []}

    time_offset = 0.0
    P_walls_final = None

    # Build stages list using durations provided
    stages = [
        ("adsorption", [0.0, float(durations['adsorption'])]),
        ("blowdown", [0.0, float(durations['blowdown'])]),
        ("heating", [0.0, float(durations['heating'])]),
        ("desorption", [0.0, float(durations['desorption'])]),
        ("pressurisation", [0.0, float(durations['pressurisation'])]),
    ]

    for cycle in range(int(n_cycles)):
        for stage_name, t_span in stages:
            # Determine pressure boundaries. If we have P_walls_final from previous stage, use that as left/right
            if P_walls_final is None:
                P_left = bed_props.get('ambient_pressure', 101325.0)
                P_right = bed_props.get('ambient_pressure', 101325.0)
            else:
                P_left = float(P_walls_final[0])
                P_right = float(P_walls_final[-1])

    # Use module's boundary definition (this creates left_values/right_values structures)
            left_values, right_values, column_direction, stage = cycle.define_boundary_conditions(stage_name, bed_props, P_left, P_right)

            stage_results = cycle.run_stage(left_values, right_values, column_direction, stage_name, t_span, current_initial_conditions, solver='BDF')

            # Unpack expected outputs (this follows the structure used in the original module)
            (stage_conditions, time_array, P_result, T_result, Tw_result,
             y1_result, y2_result, n1_result, n2_result, E_final, P_walls_result, mass_CO2_out_st,
             mass_carrier_gas_out_st, mass_CO2_in_st, mass_H2O_out_st, stage_energy) = stage_results

            # update P_walls_final for next stage
            P_walls_final = P_walls_result[:, -1]

            # store quick KPI info
            mid = column_grid['num_cells'] // 2
            cycle_profiles['time'].extend((time_array + time_offset).tolist())
            cycle_profiles['temperature'].append(float(T_result[mid, -1]))
            cycle_profiles['adsorbed_CO2'].append(float(n1_result[mid, -1]))
            cycle_profiles['wall_temperature'].append(float(Tw_result[mid, -1]))
            cycle_profiles['mass_CO2_out'].append(float(mass_CO2_out_st))
            cycle_profiles['mass_carrier_gas_out'].append(float(mass_carrier_gas_out_st))
            cycle_profiles['mass_CO2_in'].append(float(mass_CO2_in_st))
            cycle_profiles['mass_H2O_out'].append(float(mass_H2O_out_st))
            cycle_profiles['thermal_energy_input'].append(float(stage_energy[3]))
            cycle_profiles['vacuum_energy_input'].append(float(stage_energy[4]))
            cycle_profiles['fan_energy_input'].append(float(stage_energy[5]))

            # prepare next stage
            current_initial_conditions = stage_conditions
            time_offset += float(time_array[-1])

    # After cycles, compute KPIs
    # net CO2 removed — take the CO2 moles out associated with the heating stage (index 2)
    net_CO2_mols = float(cycle_profiles['mass_CO2_out'][3])
    thermal_energy = float(np.sum(cycle_profiles['thermal_energy_input']))
    vacuum_energy = float(np.sum(cycle_profiles['vacuum_energy_input']))
    fan_energy = float(np.sum(cycle_profiles['fan_energy_input']))
    total_energy = thermal_energy + vacuum_energy + fan_energy

    return {
        'net_CO2_mols': net_CO2_mols,
        'thermal_energy': thermal_energy,
        'vacuum_energy': vacuum_energy,
        'fan_energy': fan_energy,
        'total_energy': total_energy,
        'profiles': cycle_profiles
    }

# ---- Objective for optimiser ----

def objective_function(params):
    # params = [adsorption_s, blowdown_s, heating_s, pressurisation_s, T_des_K, P_vac_Pa]
    durations = {
        'adsorption': params['adsorption_time'],
        'blowdown': params['blowdown_time'],
        'heating': params['heating_time'],
        'desorption': params['desorption_time'],
        'pressurisation': params['pressurisation_time']
    }
    desorption_temp = params['desorption_temperature']
    vacuum_pressure = params['vacuum_pressure']

    sim = run_cycle_with_params(durations, desorption_temp, vacuum_pressure, n_cycles=1)

    net_CO2 = sim['net_CO2_mols']
    total_energy = sim['total_energy']

    net_CO2_scaled = net_CO2 / co2_norm
    total_energy_scaled = total_energy / energy_norm

    # Objective is to maximize CO2 capture while minimizing energy use
    obj = -net_CO2_scaled + total_energy_scaled

    # print occasional diagnostics
    objective_function.counter += 1
    if objective_function.counter % 1 == 0:
        print(f"Eval {objective_function.counter}: obj={obj:.4e}, CO2={net_CO2:.4e}, E={total_energy:.4e}, T={desorption_temp:.1f}K, Pvac={vacuum_pressure:.1f}Pa")

    return obj


def main():
    global co2_norm, energy_norm
    # We'll normalise CO2 and energy using baseline short-run values (run once)
    print("Preparing baseline evaluation to scale objective...")
    baseline_durations = {'adsorption': 13772.0, 'blowdown': 30.0, 'heating': 704.0, 'desorption': 2000.0, 'pressurisation': 50.0}
    baseline_result = run_cycle_with_params(baseline_durations, desorption_temperature=cycle.create_fixed_properties()[0]['desorption_temperature'], 
                                    vacuum_pressure=cycle.create_fixed_properties()[0]['vacuum_pressure'], n_cycles=1, verbose=False)
    if baseline_result is None:
        # fallback normalisers
        co2_norm = 1.0
        energy_norm = 1.0
        print("Baseline simulation failed — objective will use unit normalisation. Make sure dependencies are available.")
    else:
        co2_norm = max(1e-9, baseline_result['net_CO2_mols'])
        energy_norm = max(1e-9, baseline_result['total_energy'])
        print(f"Baseline net CO2 (mol)={co2_norm:.3e}, baseline energy (J)={energy_norm:.3e}")
    
    co2_norm = co2_norm
    energy_norm = energy_norm

    objective_function.counter = 0

    # ---- Bounds for (adsorption, blowdown, heating, pressurisation, T_des, P_vac) ----
    bounds = [
        (100.0, 3e4),    # adsorption time [s]
        (1.0, 100.0),    # blowdown [s]
        (100.0, 1000),    # heating [s]
        (100.0, 6000.0), # desorption [s]
        (1.0, 100.0),    # pressurisation [s]
        (323.15, 423.15),# desorption temperature [K]
        (1000.0, 101325.0) # vacuum pressure [Pa]
    ]

    print("Starting optimisation (differential evolution). ")
    start_time = time.time()
    result = differential_evolution(objective_function, bounds, maxiter=1, popsize=1, polish=True, seed=42, workers=1)
    end_time = time.time()

    print("Optimisation complete. Time elapsed: {:.1f} s".format(end_time - start_time))
    print("Result:")
    print(result)

    # Decode best solution
    best_params = result.params
    best_durations = {'adsorption': best_params[0], 'blowdown': best_params[1], 'heating': best_params[2], 'desorption': best_params[3], 'pressurisation': best_params[4]}
    best_T = best_params[5]
    best_Pvac = best_params[6]

    best_sim = run_cycle_with_params(best_durations, desorption_temperature=best_T, vacuum_pressure=best_Pvac, n_cycles=1, verbose=True)

    # Save results to JSON
    out = {
        'best_x': best_params.tolist(),
        'best_durations': best_durations,
        'best_desorption_temperature': best_T,
        'best_vacuum_pressure': best_Pvac,
        'objective_value': float(result.fun),
        'baseline_co2_norm': float(co2_norm),
        'baseline_energy_norm': float(energy_norm)
    }