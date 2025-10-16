import numpy as np
import sys

# Add the directory containing your modules to the Python path
# Adjust this path to where your files are located
sys.path.append(".")  # Current directory

# Import your modules
try:
    from additional_functions_validation import (
        create_non_uniform_grid,
        adsorption_isotherm_1,
        adsorption_isotherm_2,
    )
    import tvsa_adsorption_column_validation as column
    import run_adsorption_cycle_validation as run

    print("✓ All modules imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print(
        "Make sure all your Python files are in the current directory or adjust the sys.path.append() line"
    )
    sys.exit(1)


def create_synthetic_data():
    """
    Create synthetic experimental data for testing when real data is not available.
    This simulates what experimental data might look like.
    """
    print("Creating synthetic experimental data...")

    # Time points where "measurements" were taken
    t_temp = np.array([0, 200, 400, 600, 800, 1000, 1200, 1500, 2000, 2500, 3000])
    t_mole = np.array([0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000])

    # Synthetic temperature data (K) - starts at 292K, rises due to adsorption heat
    temp_base = 292 + 8 * (1 - np.exp(-t_temp / 800))  # Exponential rise
    temp_noise = np.random.normal(0, 0.5, len(t_temp))  # Add measurement noise
    temperature_data = temp_base + temp_noise

    # Synthetic CO2 mole fraction data - breakthrough curve
    mole_base = 0.8 * (1 - np.exp(-t_mole / 600))  # S-shaped breakthrough
    mole_noise = np.random.normal(0, 0.02, len(t_mole))  # Add measurement noise
    molefraction_data = np.clip(mole_base + mole_noise, 0, 1)  # Keep in [0,1]

    return {
        "t_temperature": t_temp,
        "temperature": temperature_data,
        "t_molefraction": t_mole,
        "molefraction": molefraction_data,
    }


def load_experimental_data():
    """
    Try to load real experimental data, fall back to synthetic if not available.
    """
    try:
        print("Attempting to load experimental data files...")

        # Try to load your CSV files
        t_observed_temperature = np.loadtxt(
            "C:\\Users\\ejm21\\Desktop\\temperature-vacuum-swing\\validation\\pini_data_temperature.csv",
            delimiter=",",
            usecols=0,
        )
        temperature_observed = np.loadtxt(
            "C:\\Users\\ejm21\\Desktop\\temperature-vacuum-swing\\validation\\pini_data_temperature.csv",
            delimiter=",",
            usecols=1,
        )
        t_observed_molefraction = np.loadtxt(
            "C:\\Users\\ejm21\\Desktop\\temperature-vacuum-swing\\validation\\pini_data_molefraction.csv",
            delimiter=",",
            usecols=0,
        )
        molefraction_observed = np.loadtxt(
            "C:\\Users\\ejm21\\Desktop\\temperature-vacuum-swing\\validation\\pini_data_molefraction.csv",
            delimiter=",",
            usecols=1,
        )

        observed_data = {
            "t_temperature": t_observed_temperature,
            "temperature": temperature_observed,
            "t_molefraction": t_observed_molefraction,
            "molefraction": molefraction_observed,
        }

        print("✓ Experimental data loaded successfully")
        print(f"  Temperature points: {len(temperature_observed)}")
        print(f"  Mole fraction points: {len(molefraction_observed)}")

        return observed_data

    except FileNotFoundError as e:
        print(f"✗ Could not load experimental data: {e}")
        print("Using synthetic data for testing...")
        return create_synthetic_data()


def test_simulation_setup():
    """
    Test that the simulation can run with default parameters.
    """
    print("\n" + "=" * 50)
    print("TESTING SIMULATION SETUP")
    print("=" * 50)

    try:
        # Create simulation conditions
        bed_properties, column_grid, initial_conditions, rtol, atol_array = (
            run.create_fixed_properties()
        )
        inlet_values, outlet_values = run.define_boundary_conditions(bed_properties)

        print("✓ Simulation properties created successfully")
        print(f"  Number of cells: {column_grid['num_cells']}")
        print(f"  Bed length: {bed_properties['bed_length']:.3f} m")
        print(f"  Initial conditions vector length: {len(initial_conditions)}")

        return True, {
            "bed_properties": bed_properties,
            "column_grid": column_grid,
            "initial_conditions": initial_conditions,
            "inlet_values": inlet_values,
            "outlet_values": outlet_values,
            "rtol": rtol,
            "atol_array": atol_array,
        }

    except Exception as e:
        print(f"✗ Error in simulation setup: {e}")
        return False, None


def test_short_simulation(simulation_conditions):
    """
    Test running a short simulation to make sure everything works.
    """
    print("\n" + "=" * 50)
    print("TESTING SHORT SIMULATION")
    print("=" * 50)

    try:
        from scipy.integrate import solve_ivp

        # Short time span for testing
        time_span = (0, 100)  # Just 100 seconds

        def ODE_func(t, results_vector):
            return column.ODE_calculations(
                t,
                results_vector=results_vector,
                column_grid=simulation_conditions["column_grid"],
                bed_properties=simulation_conditions["bed_properties"],
                inlet_values=simulation_conditions["inlet_values"],
                outlet_values=simulation_conditions["outlet_values"],
            )

        print("Running short simulation (100 seconds)...")
        output_matrix = solve_ivp(
            ODE_func,
            time_span,
            simulation_conditions["initial_conditions"],
            method="BDF",
            rtol=simulation_conditions["rtol"],
            atol=simulation_conditions["atol_array"],
        )

        if output_matrix.success:
            print("✓ Short simulation completed successfully")
            print(f"  Time points simulated: {len(output_matrix.t)}")
            print(f"  Final time: {output_matrix.t[-1]:.1f} s")
            return True
        else:
            print(f"✗ Simulation failed: {output_matrix.message}")
            return False

    except Exception as e:
        print(f"✗ Error in short simulation: {e}")
        return False


def test_parameter_fitting_class():
    """
    Test the ParameterFitter class with a simple case.
    """
    print("\n" + "=" * 50)
    print("TESTING PARAMETER FITTING CLASS")
    print("=" * 50)

    try:
        # Import the parameter fitting class
        from parameter_fitting import ParameterFitter

        print("✓ ParameterFitter class imported successfully")

        # Get simulation conditions
        success, simulation_conditions = test_simulation_setup()
        if not success:
            return False

        # Get experimental data
        observed_data = load_experimental_data()

        # Create parameter fitter
        fitter = ParameterFitter(simulation_conditions, observed_data)
        print("✓ ParameterFitter instance created successfully")

        # Test a single objective function evaluation
        print("Testing single objective function evaluation...")
        test_params = [140.0, 20.0]  # h_bed, h_wall
        param_names = ["h_bed", "h_wall"]
        weights = {"temperature": 1.0, "molefraction": 1.0}

        error = fitter.objective_function(test_params, param_names, weights)
        print(f"✓ Objective function returned error: {error:.6e}")

        if np.isfinite(error) and error > 0:
            print("✓ Error value is valid (finite and positive)")
            return True, fitter
        else:
            print(f"✗ Invalid error value: {error}")
            return False, None

    except Exception as e:
        print(f"✗ Error testing ParameterFitter: {e}")
        import traceback

        traceback.print_exc()
        return False, None


def run_quick_parameter_fitting(fitter):
    """
    Run a quick parameter fitting with few iterations for testing.
    """
    print("\n" + "=" * 50)
    print("TESTING QUICK PARAMETER FITTING")
    print("=" * 50)

    try:
        # Parameters to fit
        param_names = ["h_bed", "h_wall"]
        initial_guess = [140.0, 20.0]
        bounds = [(50, 300), (5, 50)]  # Narrower bounds for faster testing

        print(f"Fitting parameters: {param_names}")
        print(f"Initial guess: {initial_guess}")
        print(f"Bounds: {bounds}")

        # Run optimization with fewer iterations for testing
        print("Running quick optimization (limited iterations)...")

        # Modify the fitter for quick testing
        result = fitter.minimiser(
            param_names=param_names,
            initial_guess=initial_guess,
            bounds=bounds,
            weights={"temperature": 1.0, "molefraction": 1.0},
        )

        if result.success:
            print("✓ Parameter fitting completed successfully!")
            print(f"  Fitted parameters: {dict(zip(param_names, result.x))}")
            print(f"  Final error: {result.fun:.6e}")

            # Test plotting
            try:
                print("Testing result plotting...")
                fitter.plot_results()
            except Exception as e:
                print(f"⚠ Plotting failed (non-critical): {e}")

            return True
        else:
            print(f"✗ Parameter fitting failed: {result}")
            return False

    except Exception as e:
        print(f"✗ Error in parameter fitting: {e}")
        import traceback

        traceback.print_exc()
        return False


def main_test():
    """
    Main testing function that runs all tests.
    """
    print("PARAMETER FITTING TEST SUITE")
    print("=" * 60)

    # Test 1: Simulation setup
    success, simulation_conditions = test_simulation_setup()
    if not success:
        print("\n❌ SIMULATION SETUP FAILED - Cannot continue")
        return

    # Test 2: Short simulation
    if not test_short_simulation(simulation_conditions):
        print("\n❌ SHORT SIMULATION FAILED - Cannot continue")
        return

    # Test 3: Parameter fitting class
    success, fitter = test_parameter_fitting_class()
    if not success:
        print("\n❌ PARAMETER FITTING CLASS FAILED - Cannot continue")
        return

    # Test 4: Quick parameter fitting
    if run_quick_parameter_fitting(fitter):
        print("\n✅ ALL TESTS PASSED!")
        print("\nYou can now run the full parameter fitting with:")
        print("1. More iterations")
        print("2. More parameters")
        print("3. Different optimization methods")
        print("4. Your real experimental data")
    else:
        print("\n❌ PARAMETER FITTING TEST FAILED")


if __name__ == "__main__":
    main_test()
