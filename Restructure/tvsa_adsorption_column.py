"""
This is the code which describes the PDEs for the adsorption column.
The input vector is split into the vectors for each state variable, and then the inlet and outlet boundary conditions are applied.
THe bounary conditions are used to calculate the values in the ghost cells, and then the value for the state variables are
calculated at each wall position. The wall values are then used to calculate the differential term which is explorted to the
timestepper to be solved.
"""

# Import modules
import numpy as np
import matplotlib as mpl
import additional_functions as func
import config_base as base


mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["mathtext.rm"] = "serif"

### Split results vector from previous time step #######


def data_prep(results_vector, num_cells, config):
    """
        Split the results vector into individual variables.
        Four component system where y1 + y2 + y3 + y4 = 1; two adsorbing components with (CO2 and H2O)
        Vector contains P, T, Tw, y1, y2, y3, n1, n2, F
    Returns:
        --------
        tuple : (P, T, Tw, y1, y2, y3, n1, n2, F, E)
            P : Pressure [Pa]
            T : Gas temperature [K]
            Tw : Wall temperature [K]
            y1 : Mole fraction of CO2 [-]
            y2 : Mole fraction of H2O [-]
            y3 : Mole fraction of N2 [-]
            n1 : Adsorbed concentration of CO2 [mol/m³]
            n2 : Adsorbed concentration of H2O [mol/m³]
            F : Flow rate tracking variables [mol/s]
            E : Energy flow tracking variables [J/s]
    """
    # Extract state variables from the combined vector
    P = results_vector[:num_cells] * config.reference.P_ref  # Pressure
    T = (
        results_vector[num_cells : 2 * num_cells] * config.reference.T_ref
    )  # Gas temperature
    Tw = (
        results_vector[2 * num_cells : 3 * num_cells] * config.reference.T_ref
    )  # Wall temperature
    y1 = results_vector[3 * num_cells : 4 * num_cells]  # CO2 mole fraction
    y2 = results_vector[4 * num_cells : 5 * num_cells]  # H2O mole fraction
    y3 = results_vector[5 * num_cells : 6 * num_cells]  # N2 mole fraction
    y1 = np.maximum(y1, np.zeros(len(y1)))
    y2 = np.maximum(y2, np.zeros(len(y2)))
    y3 = np.maximum(y3, np.zeros(len(y3)))
    y1 = np.minimum(y1, np.ones(len(y1)))
    y2 = np.minimum(y2, np.ones(len(y2)))
    y3 = np.minimum(y3, np.ones(len(y3)))
    n1 = (
        results_vector[6 * num_cells : 7 * num_cells] * config.reference.n_ref
    )  # CO2 adsorbed concentration
    n2 = (
        results_vector[7 * num_cells : 8 * num_cells] * config.reference.n_ref
    )  # H2O adsorbed concentration
    n3 = (
        results_vector[8 * num_cells : 9 * num_cells] * config.reference.n_ref
    )  # N2 adsorbed concentration
    F = results_vector[
        9 * num_cells : 9 * num_cells + 8
    ]  # Flow tracking (8 components)
    E = results_vector[9 * num_cells + 8 :]  # Energy tracking (2 components)

    return P, T, Tw, y1, y2, y3, n1, n2, n3, F, E


def flip_column(results_vector, num_cells, config):
    """
    Flip the column state variables for the reverse simulation.
    """
    # Extract state variables from the combined vector
    P = results_vector[:num_cells] * config.reference.P_ref  # Pressure
    T = (
        results_vector[num_cells : 2 * num_cells] * config.reference.T_ref
    )  # Gas temperature
    Tw = (
        results_vector[2 * num_cells : 3 * num_cells] * config.reference.T_ref
    )  # Wall temperature
    y1 = results_vector[3 * num_cells : 4 * num_cells]  # CO2 mole fraction
    y2 = results_vector[4 * num_cells : 5 * num_cells]  # H2O mole fraction
    y3 = results_vector[5 * num_cells : 6 * num_cells]  # N2 mole fraction
    n1 = (
        results_vector[6 * num_cells : 7 * num_cells] * config.reference.n_ref
    )  # CO2 adsorbed concentration
    n2 = (
        results_vector[7 * num_cells : 8 * num_cells] * config.reference.n_ref
    )  # H2O adsorbed concentration
    n3 = (
        results_vector[8 * num_cells : 9 * num_cells] * config.reference.n_ref
    )  # N2 adsorbed concentration
    F = results_vector[
        9 * num_cells : 9 * num_cells + 8
    ]  # Flow tracking (8 components)
    E = results_vector[9 * num_cells + 8 :]

    P = P[::-1]
    T = T[::-1]
    Tw = Tw[::-1]
    y1 = y1[::-1]
    y2 = y2[::-1]
    y3 = y3[::-1]
    n1 = n1[::-1]
    n2 = n2[::-1]
    n3 = n3[::-1]
    F = F[::-1]
    E = E[::-1]  # Energy tracking (2 components)

    return P, T, Tw, y1, y2, y3, n1, n2, n3, F, E


def left_boundary_conditions(
    P, T, Tw, y1, y2, y3, column_grid, config, left_values, t=0
):
    """
    Apply inlet boundary conditions at z=0.

    Supports two inlet types:
    - "mass_flow": Specified mass flow rate with convective boundary conditions
    - "closed": Closed inlet with zero-gradient boundary conditions

    Parameters:
    -----------
    P, T, Tw, y1, y2, y3 : numpy.ndarray
        Current state variables at cell centers
    column_grid : dict
        Grid parameters including cell positions and sizes
    bed_properties : dict
        Physical properties of the bed and column
    inlet_values : dict
        Inlet operating conditions and type

    Returns:
    --------
    tuple : Boundary values and derivatives at inlet
        (P_inlet, T_inlet, Tw_inlet, y1_inlet, y2_inlet, y3_inlet,
         v_inlet, dPdz_inlet, dTwdz_inlet)
    """

    if left_values["left_type"] == "mass_flow":
        # Calculate transport properties at inlet conditions
        rho_gas_left = func.calculate_gas_density(P[0], T[0])  # [mol/m³]
        mu = func.calculate_gas_viscosity()
        v_left = left_values["velocity"]  # [m/s]
        D_m = config.transport.molecular_diffusivity
        d_p = config.adsorbent.particle_diameter
        D_l = 0.7 * D_m + 0.5 * v_left * d_p
        K_z = config.transport.K_z
        Cp_g = func.calculate_gas_heat_capacity(y1[0], y2[0], y3[0], T[0])  # [J/mol·K]
        thermal_diffusivity = K_z / (Cp_g * rho_gas_left)  # [m²/s]

        # Calculate inlet mole fractions using convective boundary conditions
        # Boundary condition: dy/dz = -(v/D_l)(y_feed - y_inlet)
        y1_left = func.quadratic_extrapolation_derivative_nonzero(
            column_grid["xCentres"][column_grid["nGhost"]],
            y1[0],
            column_grid["xCentres"][column_grid["nGhost"] + 1],
            y1[1],
            column_grid["xWalls"][column_grid["nGhost"]],
            -np.abs(v_left / D_l),
            left_values["y1_left_value"],
        )

        y2_left = func.quadratic_extrapolation_derivative_nonzero(
            column_grid["xCentres"][column_grid["nGhost"]],
            y2[0],
            column_grid["xCentres"][column_grid["nGhost"] + 1],
            y2[1],
            column_grid["xWalls"][column_grid["nGhost"]],
            -np.abs(v_left / D_l),
            left_values["y2_left_value"],
        )

        y3_left = func.quadratic_extrapolation_derivative_nonzero(
            column_grid["xCentres"][column_grid["nGhost"]],
            y3[0],
            column_grid["xCentres"][column_grid["nGhost"] + 1],
            y3[1],
            column_grid["xWalls"][column_grid["nGhost"]],
            -np.abs(v_left / D_l),
            left_values["y3_left_value"],
        )

        # Calculate average gas density for Ergun equation
        avg_density_left = (
            rho_gas_left
            / 1000
            * (
                config.fluid.MW_CO2 * y1_left
                + config.fluid.MW_H2O * y2_left
                + config.fluid.MW_N2 * y3_left
                + config.fluid.MW_O2 * (1 - y1_left - y2_left - y3_left)
            )
        )  # [kg/m³]

        # Calculate pressure drop using Ergun equation
        # dP/dz = -(150μ(1-ε)²v)/(ε³dp²) - (1.75ρ(1-ε)v²)/(ε³dp)
        dPdz_left = 1.75 * (
            1 - config.adsorbent.bed_voidage
        ) * avg_density_left * v_left**2 / (
            config.adsorbent.bed_voidage ** 3 * config.adsorbent.particle_diameter
        ) + 150 * mu * (1 - config.adsorbent.bed_voidage) ** 2 * v_left / (
            config.adsorbent.bed_voidage ** 3
            * config.adsorbent.particle_diameter ** 2
        )

        # Calculate inlet pressure from pressure gradient
        P_left = P[0] - dPdz_left * (
            column_grid["xWalls"][int(column_grid["nGhost"])]
            - column_grid["xCentres"][int(column_grid["nGhost"])]
        )

        # Calculate inlet temperature using convective boundary condition
        # dT/dz = -(v·Pe_h)(T_feed - T_inlet)
        Pe_h = config.adsorbent.bed_voidage / thermal_diffusivity  # Péclet number
        # T_inlet = (T[0] + column_grid["xCentres"][column_grid["nGhost"]] * v_inlet * Pe_h * inlet_values["feed_temperature"])/(1 + column_grid["xCentres"][column_grid["nGhost"]] * v_inlet * Pe_h)

        T_left = func.quadratic_extrapolation_derivative_nonzero(
            column_grid["xCentres"][column_grid["nGhost"]],
            T[0],
            column_grid["xCentres"][column_grid["nGhost"] + 1],
            T[1],
            column_grid["xWalls"][column_grid["nGhost"]],
            -np.abs(v_left) * config.adsorbent.bed_voidage / thermal_diffusivity,
            left_values["left_temperature"],
        )

        # Wall temperature boundary conditions
        Tw_left = Tw[0]
        dTwdz_left = 0
        dTdz_left = -(v_left * Pe_h) * (
            left_values["left_temperature"] - T_left
        )  # set inlet temperature gradient

    elif left_values["left_type"] == "pressure":
        # Fixed outlet pressure - check if it's a function or direct value

        # Extrapolate composition variables using quadratic interpolation

        y1_left = func.quadratic_extrapolation_derivative(
            column_grid["xCentres"][column_grid["nGhost"]],
            y1[0],
            column_grid["xCentres"][column_grid["nGhost"] + 1],
            y1[1],
            column_grid["xWalls"][column_grid["nGhost"]],
        )

        y2_left = func.quadratic_extrapolation_derivative(
            column_grid["xCentres"][column_grid["nGhost"]],
            y2[0],
            column_grid["xCentres"][column_grid["nGhost"] + 1],
            y2[1],
            column_grid["xWalls"][column_grid["nGhost"]],
        )

        y3_left = func.quadratic_extrapolation_derivative(
            column_grid["xCentres"][column_grid["nGhost"]],
            y3[0],
            column_grid["xCentres"][column_grid["nGhost"] + 1],
            y3[1],
            column_grid["xWalls"][column_grid["nGhost"]],
        )

        # Zero gradient for temperature
        T_left = func.quadratic_extrapolation_derivative(
            column_grid["xCentres"][column_grid["nGhost"]],
            T[0],
            column_grid["xCentres"][column_grid["nGhost"] + 1],
            T[1],
            column_grid["xWalls"][column_grid["nGhost"]],
        )

        Tw_left = Tw[0]

        P_left = func.pressure_ramp_2(t, left_values["stage"], P[0], {
            "ambient_pressure": config.operation.ambient_pressure,
            "vacuum_pressure": getattr(config.operation, "low_pressure", config.operation.ambient_pressure),
            "pressurisation_pressure": getattr(config.operation, "high_pressure", config.operation.ambient_pressure)
        })[0]
        dPdz_left = func.pressure_ramp_2(t, left_values["stage"], P[0], {
            "ambient_pressure": config.operation.ambient_pressure,
            "vacuum_pressure": getattr(config.operation, "low_pressure", config.operation.ambient_pressure),
            "pressurisation_pressure": getattr(config.operation, "high_pressure", config.operation.ambient_pressure)
        })[
            1
        ]

        if P_left is None:
            P_left = P[0] + dPdz_left * (
                column_grid["xWalls"][column_grid["nGhost"]]
                - column_grid["xCentres"][column_grid["nGhost"]]
            )
        elif dPdz_left is None:
            dPdz_left = (P_left - P[0]) / (
                column_grid["xWalls"][column_grid["nGhost"]]
                - column_grid["xCentres"][column_grid["nGhost"]]
            )

        # Calculate outlet velocity from Ergun equation
        mu = func.calculate_gas_viscosity()
        rho_gas_left = P_left / config.fluid.R / T_left  # [mol/m³]
        Cp_g = func.calculate_gas_heat_capacity(y1_left, y2_left, y3_left, T_left)
        K_z = config.transport.K_z  # [J/mol·K]
        thermal_diffusivity = K_z / (Cp_g * rho_gas_left)
        Pe_h = config.adsorbent.bed_voidage / thermal_diffusivity
        # Convert to mass density [kg/m³]
        avg_density_left = (
            rho_gas_left
            / 1000
            * (
                config.fluid.MW_CO2 * y1_left
                + config.fluid.MW_H2O * y2_left
                + config.fluid.MW_N2 * y3_left
                + config.fluid.MW_O2 * (1 - y1_left - y2_left - y3_left)
            )
        )

        # Solve Ergun equation: av² + bv + c = 0
        a = (
            1.75
            * (1 - config.adsorbent.bed_voidage)
            * avg_density_left
            / (config.adsorbent.bed_voidage ** 3 * config.adsorbent.particle_diameter)
        )
        b = (
            150
            * mu
            * (1 - config.adsorbent.bed_voidage) ** 2
            / (
                config.adsorbent.bed_voidage ** 3
                * config.adsorbent.particle_diameter ** 2
            )
        )
        c = np.abs(dPdz_left)

        v_left = -np.sign(dPdz_left) * ((-b + np.sqrt(b**2 + 4 * a * c)) / (2 * a))
        dTdz_left = -(v_left * Pe_h) * (
            left_values["left_temperature"] - T_left
        )  # set inlet temperature gradient
        dTwdz_left = 0

    elif left_values["left_type"] == "inlet_pressure":
        # Calculate pressure

        ## Ensure exactly one is provided
        P_left = func.pressure_ramp_2(t, left_values["stage"], P[0], {
            "ambient_pressure": config.operation.ambient_pressure,
            "vacuum_pressure": getattr(config.operation, "low_pressure", config.operation.ambient_pressure),
            "pressurisation_pressure": getattr(config.operation, "high_pressure", config.operation.ambient_pressure)
        })[0]
        dPdz_left = func.pressure_ramp_2(t, left_values["stage"], P[0], {
            "ambient_pressure": config.operation.ambient_pressure,
            "vacuum_pressure": getattr(config.operation, "low_pressure", config.operation.ambient_pressure),
            "pressurisation_pressure": getattr(config.operation, "high_pressure", config.operation.ambient_pressure)
        })[
            1
        ]

        if P_left is None:
            P_left = P[0] + dPdz_left * (
                column_grid["xWalls"][column_grid["nGhost"]]
                - column_grid["xCentres"][column_grid["nGhost"]]
            )
        elif dPdz_left is None:
            dPdz_left = (P_left - P[0]) / (
                column_grid["xWalls"][column_grid["nGhost"]]
                - column_grid["xCentres"][column_grid["nGhost"]]
            )

        # Calculate properties
        mu = func.calculate_gas_viscosity()
        rho_gas_left = P_left / config.fluid.R / T[0]  # [mol/m³]
        Cp_g = func.calculate_gas_heat_capacity(y1[0], y2[0], y3[0], T[0])
        K_z = config.transport.K_z  # [J/mol·K]
        thermal_diffusivity = K_z / (Cp_g * rho_gas_left)
        Pe_h = config.adsorbent.bed_voidage / thermal_diffusivity

        # Convert to mass density [kg/m³]
        avg_density_left = (
            rho_gas_left
            / 1000
            * (
                config.fluid.MW_CO2 * y1[0]
                + config.fluid.MW_H2O * y2[0]
                + config.fluid.MW_N2 * y3[0]
                + config.fluid.MW_O2 * (1 - y1[0] - y2[0] - y3[0])
            )
        )

        # Solve Ergun equation: av² + bv + c = 0
        a = (
            1.75
            * (1 - config.adsorbent.bed_voidage)
            * avg_density_left
            / (config.adsorbent.bed_voidage ** 3 * config.adsorbent.particle_diameter)
        )
        b = (
            150
            * mu
            * (1 - config.adsorbent.bed_voidage) ** 2
            / (
                config.adsorbent.bed_voidage ** 3
                * config.adsorbent.particle_diameter ** 2
            )
        )
        c = np.abs(dPdz_left)

        v_left = -np.sign(dPdz_left) * ((-b + np.sqrt(b**2 + 4 * a * c)) / (2 * a))
        # [m/s]
        #
        D_l = 0.7 * config.transport.molecular_diffusivity + 0.5 * v_left * config.adsorbent.particle_diameter
        T_left = func.quadratic_extrapolation_derivative_nonzero(
            column_grid["xCentres"][column_grid["nGhost"]],
            T[0],
            column_grid["xCentres"][column_grid["nGhost"] + 1],
            T[1],
            column_grid["xWalls"][column_grid["nGhost"]],
            -np.abs(v_left) * config.adsorbent.bed_voidage / thermal_diffusivity,
            left_values["left_temperature"],
        )

        # Boundary condition: dy/dz = -(v/D_l)(y_feed - y_inlet)
        y1_left = func.quadratic_extrapolation_derivative_nonzero(
            column_grid["xCentres"][column_grid["nGhost"]],
            y1[0],
            column_grid["xCentres"][column_grid["nGhost"] + 1],
            y1[1],
            column_grid["xWalls"][column_grid["nGhost"]],
            -np.abs(v_left / D_l),
            left_values["y1_left_value"],
        )

        y2_left = func.quadratic_extrapolation_derivative_nonzero(
            column_grid["xCentres"][column_grid["nGhost"]],
            y2[0],
            column_grid["xCentres"][column_grid["nGhost"] + 1],
            y2[1],
            column_grid["xWalls"][column_grid["nGhost"]],
            -np.abs(v_left / D_l),
            left_values["y2_left_value"],
        )

        y3_left = func.quadratic_extrapolation_derivative_nonzero(
            column_grid["xCentres"][column_grid["nGhost"]],
            y3[0],
            column_grid["xCentres"][column_grid["nGhost"] + 1],
            y3[1],
            column_grid["xWalls"][column_grid["nGhost"]],
            -np.abs(v_left / D_l),
            left_values["y3_left_value"],
        )

        dTdz_left = -(v_left * Pe_h) * (
            left_values["left_temperature"] - T_left
        )  # set inlet temperature gradient
        dTwdz_left = 0
        Tw_left = Tw[0]

    elif left_values["left_type"] == "closed":
        # Closed inlet: zero gradients for all variables
        dPdz_left = 0
        P_left = P[0]
        v_left = 0

        # Use zero-gradient extrapolation for composition variables
        y1_left = func.quadratic_extrapolation_derivative(
            column_grid["xCentres"][column_grid["nGhost"]],
            y1[0],
            column_grid["xCentres"][column_grid["nGhost"] + 1],
            y1[1],
            column_grid["xWalls"][column_grid["nGhost"]],
        )

        y2_left = func.quadratic_extrapolation_derivative(
            column_grid["xCentres"][column_grid["nGhost"]],
            y2[0],
            column_grid["xCentres"][column_grid["nGhost"] + 1],
            y2[1],
            column_grid["xWalls"][column_grid["nGhost"]],
        )

        y3_left = func.quadratic_extrapolation_derivative(
            column_grid["xCentres"][column_grid["nGhost"]],
            y3[0],
            column_grid["xCentres"][column_grid["nGhost"] + 1],
            y3[1],
            column_grid["xWalls"][column_grid["nGhost"]],
        )

        T_left = func.quadratic_extrapolation_derivative(
            column_grid["xCentres"][column_grid["nGhost"]],
            T[0],
            column_grid["xCentres"][column_grid["nGhost"] + 1],
            T[1],
            column_grid["xWalls"][column_grid["nGhost"]],
        )

        dTdz_left = 0
        Tw_left = Tw[0]
        dTwdz_left = 0

    return (
        P_left,
        T_left,
        Tw_left,
        y1_left,
        y2_left,
        y3_left,
        v_left,
        dPdz_left,
        dTwdz_left,
        dTdz_left,
    )


def right_boundary_conditions(
    P, T, Tw, y1, y2, y3, column_grid, config, right_values, t=0
):
    """
    Apply outlet boundary conditions at z=L.

    Supports three outlet types:
    - "pressure": Fixed pressure with extrapolated composition
    - "closed": Closed outlet with zero-gradient boundary conditions
    - "mass_flow": Fixed mass flow rate (implementation incomplete)

    Parameters:
    -----------
    P, T, Tw, y1, y2, y3 : numpy.ndarray
        Current state variables at cell centers
    column_grid : dict
        Grid parameters
    bed_properties : dict
        Physical properties
    right_values : dict
        Outlet operating conditions and type

    Returns:
    --------
    tuple : Boundary values and derivatives at outlet
        (P_right, T_right, Tw_right, y1_right, y2_right, y3_right,
         v_right, dPdz_right, dTwdz_right)
    """

    if right_values["right_type"] == "pressure":
        # Fixed outlet pressure - check if it's a function or direct value

        # Extrapolate composition variables using quadratic interpolation

        y1_right = func.quadratic_extrapolation_derivative(
            column_grid["xCentres"][-(column_grid["nGhost"] + 1)],
            y1[-1],
            column_grid["xCentres"][-(column_grid["nGhost"] + 2)],
            y1[-2],
            column_grid["xWalls"][-(column_grid["nGhost"] + 1)],
        )

        y2_right = func.quadratic_extrapolation_derivative(
            column_grid["xCentres"][-(column_grid["nGhost"] + 1)],
            y2[-1],
            column_grid["xCentres"][-(column_grid["nGhost"] + 2)],
            y2[-2],
            column_grid["xWalls"][-(column_grid["nGhost"] + 1)],
        )

        y3_right = func.quadratic_extrapolation_derivative(
            column_grid["xCentres"][-(column_grid["nGhost"] + 1)],
            y3[-1],
            column_grid["xCentres"][-(column_grid["nGhost"] + 2)],
            y3[-2],
            column_grid["xWalls"][-(column_grid["nGhost"] + 1)],
        )

        # Zero gradient for temperature
        T_right = func.quadratic_extrapolation_derivative(
            column_grid["xCentres"][-(column_grid["nGhost"] + 1)],
            T[-1],
            column_grid["xCentres"][-(column_grid["nGhost"] + 2)],
            T[-2],
            column_grid["xWalls"][-(column_grid["nGhost"] + 1)],
        )

        Tw_right = Tw[-1]

        ## Ensure exactly one is provided
        P_right = func.pressure_ramp_2(t, right_values["stage"], P[-1], {
            "ambient_pressure": config.operation.ambient_pressure,
            "vacuum_pressure": getattr(config.operation, "low_pressure", config.operation.ambient_pressure),
            "pressurisation_pressure": getattr(config.operation, "high_pressure", config.operation.ambient_pressure)
        })[
            0
        ]
        dPdz_right = func.pressure_ramp_2(
            t, right_values["stage"], P[-1], {
            "ambient_pressure": config.operation.ambient_pressure,
            "vacuum_pressure": getattr(config.operation, "low_pressure", config.operation.ambient_pressure),
            "pressurisation_pressure": getattr(config.operation, "high_pressure", config.operation.ambient_pressure)
        }
        )[1]

        if P_right is None:
            P_right = P[-1] + dPdz_right * (
                column_grid["xWalls"][-(column_grid["nGhost"] + 1)]
                - column_grid["xCentres"][-(column_grid["nGhost"] + 1)]
            )
        elif dPdz_right is None:
            dPdz_right = (P_right - P[-1]) / (
                column_grid["xWalls"][-(column_grid["nGhost"] + 1)]
                - column_grid["xCentres"][-(column_grid["nGhost"] + 1)]
            )

        # Calculate outlet velocity from Ergun equation
        mu = func.calculate_gas_viscosity()
        rho_gas_right = P_right / config.fluid.R / T_right  # [mol/m³]

        # Convert to mass density [kg/m³]
        avg_density_right = (
            rho_gas_right
            / 1000
            * (
                config.fluid.MW_CO2 * y1_right
                + config.fluid.MW_H2O * y2_right
                + config.fluid.MW_N2 * y3_right
                + config.fluid.MW_O2 * (1 - y1_right - y2_right - y3_right)
            )
        )

        # Calculate pressure gradient
        dPdz_right = (P_right - P[-1]) / (
            column_grid["xWalls"][-(column_grid["nGhost"] + 1)]
            - column_grid["xCentres"][-(column_grid["nGhost"] + 1)]
        )

        # Solve Ergun equation: av² + bv + c = 0
        a = (
            1.75
            * (1 - config.adsorbent.bed_voidage)
            * avg_density_right
            / (config.adsorbent.bed_voidage ** 3 * config.adsorbent.particle_diameter)
        )
        b = (
            150
            * mu
            * (1 - config.adsorbent.bed_voidage) ** 2
            / (
                config.adsorbent.bed_voidage ** 3
                * config.adsorbent.particle_diameter ** 2
            )
        )
        c = np.abs(dPdz_right)

        v_right = -np.sign(dPdz_right) * ((-b + np.sqrt(b**2 + 4 * a * c)) / (2 * a))
        dTwdz_right = 0
        dTdz_right = 0

    elif right_values["right_type"] == "closed":
        # Closed outlet: zero gradients

        P_right = P[-1]

        y1_right = func.quadratic_extrapolation_derivative(
            column_grid["xCentres"][-(column_grid["nGhost"] + 1)],
            y1[-1],
            column_grid["xCentres"][-(column_grid["nGhost"] + 2)],
            y1[-2],
            column_grid["xWalls"][-(column_grid["nGhost"] + 1)],
        )

        y2_right = func.quadratic_extrapolation_derivative(
            column_grid["xCentres"][-(column_grid["nGhost"] + 1)],
            y2[-1],
            column_grid["xCentres"][-(column_grid["nGhost"] + 2)],
            y2[-2],
            column_grid["xWalls"][-(column_grid["nGhost"] + 1)],
        )

        y3_right = func.quadratic_extrapolation_derivative(
            column_grid["xCentres"][-(column_grid["nGhost"] + 1)],
            y3[-1],
            column_grid["xCentres"][-(column_grid["nGhost"] + 2)],
            y3[-2],
            column_grid["xWalls"][-(column_grid["nGhost"] + 1)],
        )

        T_right = func.quadratic_extrapolation_derivative(
            column_grid["xCentres"][-(column_grid["nGhost"] + 1)],
            T[-1],
            column_grid["xCentres"][-(column_grid["nGhost"] + 2)],
            T[-2],
            column_grid["xWalls"][-(column_grid["nGhost"] + 1)],
        )

        Tw_right = Tw[-1]
        v_right = 0
        dPdz_right = 0
        dTwdz_right = 0
        dTdz_right = 0

    elif right_values["right_type"] == "mass_flow":
        # Mass flow outlet (implementation incomplete)
        P_right = P[-1]
        # TODO: Implement mass flow boundary condition
        # Need to specify other variables based on mass flow constraint

    return (
        P_right,
        T_right,
        Tw_right,
        y1_right,
        y2_right,
        y3_right,
        v_right,
        dPdz_right,
        dTwdz_right,
        dTdz_right,
    )


def ghost_cell_calculations(
    P,
    T,
    Tw,
    y1,
    y2,
    y3,
    P_left,
    P_right,
    T_left,
    T_right,
    Tw_left,
    Tw_right,
    y1_left,
    y1_right,
    y2_left,
    y2_right,
    y3_left,
    y3_right,
    column_grid,
):
    """
    Calculate ghost cell values using boundary conditions.

    Ghost cells are fictitious cells outside the computational domain used to
    implement boundary conditions in finite volume methods.

    Parameters:
    -----------
    P, T, Tw, y1, y2, y3 : numpy.ndarray
        State variables at interior cell centers
    *_inlet, *_right : float
        Boundary values at inlet and outlet
    column_grid : dict
        Grid parameters

    Returns:
    --------
    tuple : Extended arrays including ghost cells
        (P_all, T_all, Tw_all, y1_all, y2_all, y3_all)
    """

    # Pressure ghost cells using linear extrapolation
    P_ghost_left = P[0] + (column_grid["xCentres"][0] - column_grid["xCentres"][1]) * (
        P[0] - P_left
    ) / (
        column_grid["xCentres"][column_grid["nGhost"]]
        - column_grid["xWalls"][column_grid["nGhost"]]
    )

    P_ghost_right = P[-1] - (
        column_grid["xCentres"][-1]
        - column_grid["xCentres"][-column_grid["nGhost"] - 1]
    ) * (P[-1] - P_right) / (
        column_grid["xCentres"][-(column_grid["nGhost"] + 1)]
        - column_grid["xWalls"][-(column_grid["nGhost"] + 1)]
    )

    P_all = np.concatenate([np.array([P_ghost_left]), P, np.array([P_ghost_right])])

    # Composition ghost cells using quadratic extrapolation
    y1_ghost_left = y1[0] + (
        column_grid["xCentres"][0] - column_grid["xCentres"][1]
    ) * (y1[0] - y1_left) / (
        column_grid["xCentres"][column_grid["nGhost"]]
        - column_grid["xWalls"][column_grid["nGhost"]]
    )

    y1_ghost_right = y1[-1] - (
        column_grid["xCentres"][-1]
        - column_grid["xCentres"][-column_grid["nGhost"] - 1]
    ) * (y1[-1] - y1_right) / (
        column_grid["xCentres"][-(column_grid["nGhost"] + 1)]
        - column_grid["xWalls"][-(column_grid["nGhost"] + 1)]
    )

    y1_all = np.concatenate((np.array([y1_ghost_left]), y1, np.array([y1_ghost_right])))

    # Similar calculations for y2 and y3
    y2_ghost_left = y2[0] + (
        column_grid["xCentres"][0] - column_grid["xCentres"][1]
    ) * (y2[0] - y2_left) / (
        column_grid["xCentres"][column_grid["nGhost"]]
        - column_grid["xWalls"][column_grid["nGhost"]]
    )

    y2_ghost_right = y2[-1] - (
        column_grid["xCentres"][-1]
        - column_grid["xCentres"][-column_grid["nGhost"] - 1]
    ) * (y2[-1] - y2_right) / (
        column_grid["xCentres"][-(column_grid["nGhost"] + 1)]
        - column_grid["xWalls"][-(column_grid["nGhost"] + 1)]
    )

    y2_all = np.concatenate((np.array([y2_ghost_left]), y2, np.array([y2_ghost_right])))

    y3_ghost_left = y3[0] + (
        column_grid["xCentres"][0] - column_grid["xCentres"][1]
    ) * (y3[0] - y3_left) / (
        column_grid["xCentres"][column_grid["nGhost"]]
        - column_grid["xWalls"][column_grid["nGhost"]]
    )

    y3_ghost_right = y3[-1] - (
        column_grid["xCentres"][-1]
        - column_grid["xCentres"][-column_grid["nGhost"] - 1]
    ) * (y3[-1] - y3_right) / (
        column_grid["xCentres"][-(column_grid["nGhost"] + 1)]
        - column_grid["xWalls"][-(column_grid["nGhost"] + 1)]
    )

    y3_all = np.concatenate((np.array([y3_ghost_left]), y3, np.array([y3_ghost_right])))

    # Temperature ghost cells

    T_ghost_left = T[0] + (column_grid["xCentres"][0] - column_grid["xCentres"][1]) * (
        T[0] - T_left
    ) / (
        column_grid["xCentres"][column_grid["nGhost"]]
        - column_grid["xWalls"][column_grid["nGhost"]]
    )

    T_ghost_right = T[-1] - (
        column_grid["xCentres"][-1]
        - column_grid["xCentres"][-column_grid["nGhost"] - 1]
    ) * (T[-1] - T_right) / (
        column_grid["xCentres"][-(column_grid["nGhost"] + 1)]
        - column_grid["xWalls"][-(column_grid["nGhost"] + 1)]
    )

    T_all = np.concatenate((np.array([T_ghost_left]), T, np.array([T_ghost_right])))

    # Wall temperature ghost cells
    Tw_ghost_left = Tw[0] + (
        column_grid["xCentres"][0] - column_grid["xCentres"][1]
    ) * (Tw[0] - Tw_left) / (
        column_grid["xCentres"][column_grid["nGhost"]]
        - column_grid["xWalls"][column_grid["nGhost"]]
    )

    Tw_ghost_right = Tw[-1] - (
        column_grid["xCentres"][-1]
        - column_grid["xCentres"][-column_grid["nGhost"] - 1]
    ) * (Tw[-1] - Tw_right) / (
        column_grid["xCentres"][-(column_grid["nGhost"] + 1)]
        - column_grid["xWalls"][-(column_grid["nGhost"] + 1)]
    )

    Tw_all = np.concatenate([np.array([Tw_ghost_left]), Tw, np.array([Tw_ghost_right])])

    return P_all, T_all, Tw_all, y1_all, y2_all, y3_all


def calculate_internal_wall_values(
    P_all,
    T_all,
    Tw_all,
    y1_all,
    y2_all,
    y3_all,
    P_left,
    P_right,
    T_left,
    T_right,
    Tw_left,
    Tw_right,
    y1_left,
    y1_right,
    y2_left,
    y2_right,
    y3_left,
    y3_right,
    v_left,
    v_right,
    dPdz_left,
    dPdz_right,
    dTwdz_left,
    dTwdz_right,
    dTdz_left,
    dTdz_right,
    config,
    column_grid,
    left_values,
):
    def van_leer(state_variable):
        epsilon = 1.0e-10
        Nx = int(column_grid["num_cells"])
        # y1 vector at cell walls, from van leer flux limiter function
        R_r = (
            column_grid["deltaZ"][2 : Nx + 2] + column_grid["deltaZ"][1 : Nx + 1]
        ) / column_grid["deltaZ"][1 : Nx + 1]
        r_r1 = (
            ((state_variable[1 : Nx + 1] - state_variable[:Nx]) + epsilon)
            / ((state_variable[2 : Nx + 2] - state_variable[1 : Nx + 1]) + epsilon)
            * (column_grid["deltaZ"][2 : Nx + 2] + column_grid["deltaZ"][1 : Nx + 1])
            / (column_grid["deltaZ"][1 : Nx + 1] + column_grid["deltaZ"][0:Nx])
        )
        modified_van_leer = (0.5 * R_r * r_r1 + 0.5 * R_r * abs(r_r1)) / (
            R_r + r_r1 - 1
        )
        flux_limiter = modified_van_leer / R_r
        state_var_walls = state_variable[1 : Nx + 1] + flux_limiter * (
            state_variable[2 : Nx + 2] - state_variable[1 : Nx + 1]
        )
        return state_var_walls

    y1_walls = van_leer(y1_all)
    y1_walls[-1] = y1_right
    y1_walls = np.concatenate((np.array([y1_left]), y1_walls))

    y2_walls = van_leer(y2_all)
    y2_walls[-1] = y2_right
    y2_walls = np.concatenate((np.array([y2_left]), y2_walls))

    y3_walls = van_leer(y3_all)
    y3_walls[-1] = y3_right
    y3_walls = np.concatenate((np.array([y3_left]), y3_walls))

    T_walls = van_leer(T_all)
    T_walls[-1] = T_right
    T_walls = np.concatenate((np.array([T_left]), T_walls))

    Nx = int(column_grid["num_cells"])

    dTdz_walls = np.array(
        (T_all[1 : Nx + 2] - T_all[0 : Nx + 1])
        / (column_grid["xCentres"][1 : Nx + 2] - column_grid["xCentres"][0 : Nx + 1])
    )
    dTdz_walls[-1] = dTdz_right
    dTdz_walls[0] = dTdz_left

    # calculate dP/dz at internal cell walls by linear interpolation
    dPdz_walls = np.array(
        (P_all[1 : Nx + 2] - P_all[0 : Nx + 1])
        / (column_grid["xCentres"][1 : Nx + 2] - column_grid["xCentres"][0 : Nx + 1])
    )
    dPdz_walls[0] = dPdz_left  # set inlet pressure gradient
    dPdz_walls[-1] = dPdz_right

    # calculate P at cell walls by interpolation
    P_walls = np.array(
        P_all[0 : Nx + 1] + dPdz_walls * (column_grid["deltaZ"][0 : Nx + 1] / 2)
    )
    P_walls[0] = P_left  # set inlet pressure
    P_walls[-1] = P_right  # set outlet pressure

    # calculate dTw/dz at internal cell walls by linear interpolation
    dTwdz_walls = np.array(
        (Tw_all[1 : Nx + 2] - Tw_all[0 : Nx + 1])
        / (column_grid["xCentres"][1 : Nx + 2] - column_grid["xCentres"][0 : Nx + 1])
    )
    dTwdz_walls[0] = dTwdz_left  # set inlet pressure gradient
    dTwdz_walls[-1] = dTwdz_right

    # calculate Tw at cell walls by interpolation
    Tw_walls = np.array(
        Tw_all[0 : Nx + 1] + dTwdz_walls * (column_grid["deltaZ"][0 : Nx + 1] / 2)
    )
    Tw_walls[0] = Tw_left  # set inlet pressure
    Tw_walls[-1] = Tw_right  # set outlet pressure

    rho_gas_walls = (
        P_walls / config.fluid.R / T_walls
    )  # Assuming ideal gas law for density calculation
    avg_density_walls = (
        rho_gas_walls
        / 1000
        * (
            config.fluid.MW_CO2 * y1_walls
            + config.fluid.MW_H2O * (y2_walls)
            + config.fluid.MW_N2 * (y3_walls)
            + config.fluid.MW_O2 * (1 - y1_walls - y2_walls - y3_walls)
        )
    )
    mu = func.calculate_gas_viscosity()
    a = (
        1.75
        * (1 - config.adsorbent.bed_voidage)
        * avg_density_walls
        / (config.adsorbent.bed_voidage ** 3 * config.adsorbent.particle_diameter)
    )
    b = (
        150
        * mu
        * (1 - config.adsorbent.bed_voidage) ** 2
        / (
            config.adsorbent.bed_voidage ** 3
            * config.adsorbent.particle_diameter ** 2
        )
    )
    c = np.abs(dPdz_walls[:])

    dominant = b**2 + 4 * a * c

    if np.any(dominant < 0):
        print("Final P_walls values before error:", P_walls)
        raise ValueError(
            "Negative value under square root in velocity calculation. Check your inputs and boundary conditions."
        )
    v_walls = np.array((-b + np.sqrt(dominant)) / (2 * a))  # should have N + 1 values
    v_walls = np.multiply(
        -np.sign(dPdz_walls), v_walls
    )  # make sure velocity is in the correct direction
    v_walls[0] = v_left  # set inlet velocity
    v_walls[-1] = v_right  # set outlet velocity

    return (
        P_walls,
        T_walls,
        Tw_walls,
        y1_walls,
        y2_walls,
        y3_walls,
        v_walls,
        dTdz_walls,
        dTwdz_walls,
    )


def ODE_calculations(
    t,
    results_vector,
    column_grid,
    config,
    left_values,
    right_values,
    column_direction,
    stage,
    P_initial,
    config_module
):
    """
    Calculate the ODEs for the adsorption column model.

    This function computes the derivatives of the state variables including:
    - Pressure (P)
    - Temperature (T)
    - Wall temperature (Tw)
    - Component mole fractions (y1, y2, y3)
    - Adsorbed amounts (n1, n2)
    - Flow rates (F) and energy (E) for mass/energy balance tracking

    Parameters:
    -----------
    t : float
        Time variable
    results_vector : array
        Current state vector containing all variables
    column_grid : dict
        Grid properties including number of cells and spacing
    bed_properties : dict
        Physical properties of the bed and materials
    inlet_values : dict
        Inlet boundary conditions
    right_values : dict
        Outlet boundary conditions

    Returns:
    --------
    array
        Derivatives vector for all state variables
    """
    num_cells = column_grid["num_cells"]

    # Split results vector into individual variables

    if column_direction == "forwards":
        P, T, Tw, y1, y2, y3, n1, n2, n3, F, E = data_prep(
            results_vector, num_cells, config
        )
    elif column_direction == "reverse":
        P, T, Tw, y1, y2, y3, n1, n2, n3, F, E = flip_column(
            results_vector, num_cells, config
        )

    # =========================================================================
    # BOUNDARY CONDITIONS AND GHOST CELLS
    # =========================================================================

    # Apply inlet boundary conditions
    (
        P_left,
        T_left,
        Tw_left,
        y1_left,
        y2_left,
        y3_left,
        v_left,
        dPdz_left,
        dTwdz_left,
        dTdz_left,
    ) = left_boundary_conditions(
        P, T, Tw, y1, y2, y3, column_grid, config, left_values, t=t
    )

    # Apply outlet boundary conditions
    (
        P_right,
        T_right,
        Tw_right,
        y1_right,
        y2_right,
        y3_right,
        v_right,
        dPdz_right,
        dTwdz_right,
        dTdz_right,
    ) = right_boundary_conditions(
        P, T, Tw, y1, y2, y3, column_grid, config, right_values, t=t
    )

    # Calculate ghost cell values
    P_all, T_all, Tw_all, y1_all, y2_all, y3_all = ghost_cell_calculations(
        P,
        T,
        Tw,
        y1,
        y2,
        y3,
        P_left,
        P_right,
        T_left,
        T_right,
        Tw_left,
        Tw_right,
        y1_left,
        y1_right,
        y2_left,
        y2_right,
        y3_left,
        y3_right,
        column_grid,
    )

    # Calculate internal wall values
    (
        P_walls,
        T_walls,
        Tw_walls,
        y1_walls,
        y2_walls,
        y3_walls,
        v_walls,
        dTdz_walls,
        dTwdz_walls,
    ) = calculate_internal_wall_values(
        P_all,
        T_all,
        Tw_all,
        y1_all,
        y2_all,
        y3_all,
        P_left,
        P_right,
        T_left,
        T_right,
        Tw_left,
        Tw_right,
        y1_left,
        y1_right,
        y2_left,
        y2_right,
        y3_left,
        y3_right,
        v_left,
        v_right,
        dPdz_left,
        dPdz_right,
        dTwdz_left,
        dTwdz_right,
        dTdz_left,
        dTdz_right,
        config,
        column_grid,
        left_values,
    )

    walls = np.concatenate(
        (P_walls, T_walls, Tw_walls, y1_walls, y2_walls, y3_walls, v_walls)
    )

    # =========================================================================
    # MASS TRANSFER AND ADSORPTION KINETICS
    # =========================================================================

    # Mass transfer constants
    k1 = config.transport.mass_transfer_CO2
    k2 = config.transport.mass_transfer_H2O
    k3 = config.transport.mass_transfer_N2

    # Solid phase balance for adsorbed components
    # ∂q₁/∂t = k₁(q₁* - q₁)
    n1_eq, deltaH_1 = base.calculate_CO2_loading(config, P, T, y1, y2, y3)
    dn1dt = k1 * (n1_eq - n1)

    # ∂q₂/∂t = k₂(q₂* - q₂)
    n2_eq, deltaH_2 = base.calculate_H2O_loading(config, P, T, y1, y2, y3)
    dn2dt = k2 * (n2_eq - n2)

    n3_eq, deltaH_3 = base.calculate_N2_loading(config, P, T, y1, y2, y3)
    dn3dt = k3 * (n3_eq - n3)

    # =========================================================================
    # TRANSPORT PROPERTIES
    # =========================================================================

    # Heat capacities
    Cp_g = func.calculate_gas_heat_capacity(y1, y2, y3, T)  # J/(mol*K)
    
    Cp_solid = config.adsorbent.sorbent_heat_capacity  # J/(kg*K)
    Cp_1 = Cp_g  # bed_properties["heat_capacity_1"] # J/(mol*K)
    Cp_2 = Cp_g  # bed_properties["heat_capacity_2"] # J/(mol*K)
    Cp_3 = Cp_g  # bed_properties["heat_capacity_3"] # J/(mol*K)
    Cp_4 = Cp_g  # bed_properties["heat_capacity_4"] # J/(mol*K)

    # Thermal conductivities and heat transfer coefficients
    K_z = config.transport.K_z
    K_wall = config.transport.K_wall
    h_bed, h_wall = (
        config.transport.h_bed,
        config.transport.h_wall,
    )  # Heat transfer coefficients (W/m²*K)

    # Axial dispersion coefficient
    D_l = 0.7 * config.transport.molecular_diffusivity + 0.5 * v_left * config.adsorbent.particle_diameter

    # =========================================================================
    # ENERGY BALANCE - COLUMN TEMPERATURE
    # =========================================================================

    a_2 = (
        config.adsorbent.total_voidage * P / (config.fluid.R * T) * Cp_g
        + config.adsorbent.bed_density * Cp_solid
        + (Cp_1 * n1 + Cp_2 * n2 + Cp_3 * n3)
        - config.adsorbent.total_voidage * P / T
    )
    #  a_2 = \varepsilon_t \frac{P}{RT}C_{p,g}+\rho_b C_{p,s}+(1-\varepsilon_b)C_{p,i} \ n_i - \varepsilon_t \frac{P}{T}
    # Column energy balance terms
    conduction_term = (
        +K_z
        * (1 / column_grid["deltaZ"][1:-1])
        * (dTdz_walls[1 : num_cells + 1] - dTdz_walls[:num_cells])
    )

    convection_term = (
        -Cp_g
        / config.fluid.R
        / column_grid["deltaZ"][1:-1]
        * (
            v_walls[1 : num_cells + 1] * P_walls[1 : num_cells + 1]
            - v_walls[:num_cells] * P_walls[:num_cells]
        )
    )

    accumulation_term = +config.fluid.R * T  * (dn1dt + dn2dt) 
    - T / (
        column_grid["deltaZ"][1:-1]
    ) * (
        P_walls[1 : num_cells + 1]
        * v_walls[1 : num_cells + 1]
        / T_walls[1 : num_cells + 1]
        - P_walls[:num_cells] * v_walls[:num_cells] / T_walls[:num_cells]
    )

    adsorption_heat_term = + (
        (np.abs(deltaH_1)) * dn1dt + (np.abs(deltaH_2)) * dn2dt + (np.abs(deltaH_3)) * dn3dt
    )

    adsorbent_heat_term = (
        - T * (Cp_1 * dn1dt + Cp_2 * dn2dt + Cp_3 * dn3dt)
    )

    heat_transfer_term = -2 * h_bed * (T - Tw) / (config.geometry.inner_bed_radius)

    # Combined temperature derivative
    dTdt = (
        1
        / a_2
        * (
            conduction_term
            + convection_term
            + accumulation_term
            + adsorption_heat_term
            + adsorbent_heat_term
            + heat_transfer_term
        )
    )

#     \frac{\partial T}{\partial t} = \frac{1}{a_2}\left[
#  K_z \ \varepsilon_b\frac{\partial}{\partial z}\left(\frac{\partial T}{\partial z}\right) - \frac{C_{p,g} \ \varepsilon_b}{R}\frac{\partial}{\partial z} \left(P \ v \right) \right.\\
#   - (1-\varepsilon_b){RT} \left(\frac{\partial n_1}{\partial t}+\frac{\partial n_2}{\partial t}\right) - T \ \varepsilon_b \frac{\partial}{\partial z} \left( \frac{P \ v}{T} \right)  \\ 
# \left. + (1-\varepsilon_b) \left(  \Delta H_1 \frac{\partial q_1}{\partial t} +\Delta H_2 \frac{\partial q_2}{\partial t}\right)
#  + (1-\varepsilon_b)  C_{p,ads} T \left( \frac{\partial q_1}{\partial t} + \frac{\partial q_2}{\partial t}\right) + \frac{2h_{bed}}{R_{in}}(T-T_w) \right]
#     # =========================================================================
    # TOTAL ENERGY BALANCE - UGLY VERSION
    # =========================================================================
    """
    a_2 = (
        + bed_properties["bed_density"] * Cp_solid
        + (1 - bed_properties["bed_voidage"]) * (Cp_1 * n1 + Cp_2 * n2)
        + Cp_g *  bed_properties["total_voidage"] * P / (bed_properties["R"] * T)
        - bed_properties["total_voidage"] * P / T
    )

    # Column energy balance terms


    convection_term = - (bed_properties["bed_voidage"] * 1 / column_grid["deltaZ"][1:-1] 
                         * (P_walls[1:num_cells+1] * v_walls[1:num_cells+1] / bed_properties["R"] - 
                            P_walls[:num_cells] * v_walls[:num_cells] / bed_properties["R"]))
    
    kinetic_term = + ( bed_properties["bed_voidage"] * 1/ column_grid["deltaZ"][1:-1] *
                    (P_walls[1:num_cells+1] * v_walls[1:num_cells+1] - 
                     P_walls[:num_cells] * v_walls[:num_cells]))
    
    conduction_term = + ( bed_properties["bed_voidage"] * K_z / column_grid["deltaZ"][1:-1] *
                        (dTdz_walls[1:num_cells+1] - dTdz_walls[:num_cells]))
    
    dy1dz = (y1_all[1:num_cells+2]- y1_all[0:num_cells+1]) / (column_grid["xCentres"][1:num_cells+2] - column_grid["xCentres"][0:num_cells+1])
    dy2dz = (y2_all[1:num_cells+2]- y2_all[0:num_cells+1]) / (column_grid["xCentres"][1:num_cells+2] - column_grid["xCentres"][0:num_cells+1])
    dy3dz = (y3_all[1:num_cells+2]- y3_all[0:num_cells+1]) / (column_grid["xCentres"][1:num_cells+2] - column_grid["xCentres"][0:num_cells+1])
    dy4dz = - dy1dz - dy2dz - dy3dz

    dispersion_term = + ( bed_properties["bed_voidage"] * D_l / bed_properties["R"] *
                          ( 1 / column_grid["deltaZ"][1:-1] * 
                           (P_walls[1:num_cells+1] * (Cp_1 *dy1dz[1:num_cells+1] + Cp_2 * dy2dz[1:num_cells+1] + 
                                                      Cp_3 * dy3dz[1:num_cells+1] + Cp_4 * dy4dz[1:num_cells+1])
                            - P_walls[:num_cells] * (Cp_1 * dy1dz[:num_cells] + Cp_2 * dy2dz[:num_cells] + 
                                                                                 Cp_3 * dy3dz[:num_cells] + Cp_4 * dy4dz[:num_cells]))
    ))

    adsorbent_heat_generation = + (1-bed_properties["bed_voidage"]) * (np.abs(deltaH_1) * dn1dt + np.abs(deltaH_2) * dn2dt)

    accumulation_term = - ((1-bed_properties["bed_voidage"])
                           * bed_properties["R"] * T * (dn1dt + dn2dt)

                           + bed_properties["bed_voidage"] 
                           * T / column_grid["deltaZ"][1:-1] *
                           (P_walls[1:num_cells+1] * v_walls[1:num_cells+1] / T_walls[1:num_cells+1] -
                            P_walls[:num_cells] * v_walls[:num_cells] / T_walls[:num_cells]))
    
    heat_transfer_term = (-2 * h_bed * (T - Tw) / 
                         (bed_properties["inner_bed_radius"]))
    
    dTdt = (1 / a_2 * (conduction_term + convection_term + kinetic_term + dispersion_term +
                      adsorbent_heat_generation + accumulation_term + heat_transfer_term))
    """
    # =========================================================================
    # TOTAL MASS BALANCE - PRESSURE
    # =========================================================================

    thermal_expansion_term = P / T * dTdt
    adsorption_term = (
        - 1
        / config.adsorbent.total_voidage
        * config.fluid.R
        * T
        * (dn1dt + dn2dt + dn3dt)
    )
    convective_term = (
        -T
        / config.adsorbent.total_voidage
        * 1
        / column_grid["deltaZ"][1:-1]
        * (
            P_walls[1 : num_cells + 1]
            * v_walls[1 : num_cells + 1]
            / T_walls[1 : num_cells + 1]
            - P_walls[:num_cells] * v_walls[:num_cells] / T_walls[:num_cells]
        )
    )

    dPdt = thermal_expansion_term + adsorption_term + convective_term

    # \dfrac{\partial P}{\partial t} = \dfrac{P}{T}\dfrac{\partial T}{\partial t} - \dfrac{(1-\varepsilon)}{\varepsilon}{RT} \left(\frac{\partial q_1}{\partial t}+\frac{\partial q_2}{\partial t}\right)
    # - \frac{1}{\varepsilon} \frac{T}{\Delta z} \left( \frac{P_{i+1/2} \ v_{i+1/2}}{T_{i+1/2}}-\frac{P_{i-1/2} \ v_{i-1/2}}{T_{i-1/2}} \right)"""

    # Non-distcretized
#     \dfrac{\partial P}{\partial t} = \dfrac{P}{T}\dfrac{\partial T}{\partial t} - \dfrac{(1-\varepsilon_b)}{\varepsilon_t}{RT} \left(\frac{\partial q_1}{\partial t}+\frac{\partial q_2}{\partial t}\right)
#  - \frac{\varepsilon_b}{\varepsilon_t} T \dfrac{\partial}{\partial z}\left( \frac{P \ v}{T} \right)

    # =========================================================================
    # WALL ENERGY BALANCE
    # =========================================================================

    # Wall temperature second derivative
    d2Twdt2 = (
        1
        / column_grid["deltaZ"][1:-1]
        * (dTwdz_walls[1 : num_cells + 1] - dTwdz_walls[:num_cells])
    )

    wall_conduction = K_wall * d2Twdt2
    bed_heat_exchange = (
        2
        * config.geometry.inner_bed_radius
        * h_bed
        * (T - Tw)
        / (
            config.geometry.outer_bed_radius ** 2
            - config.geometry.inner_bed_radius ** 2
        )
    )
    ambient_heat_loss = (
        -2
        * config.geometry.outer_bed_radius
        * h_wall
        * (Tw - config.operation.outside_temperature)
        / (
            config.geometry.outer_bed_radius ** 2
            - config.geometry.inner_bed_radius ** 2
        )
    )

    dTwdt = (
        1
        / (config.geometry.wall_heat_capacity * config.geometry.wall_density)
        * (wall_conduction + bed_heat_exchange + ambient_heat_loss)
    )
    dTwdt = 0 * dTwdt  # Disable wall energy balance for now
    # """\frac{\partial T_w}{\partial t} =
    # \frac{1}{\rho_w \ C_{p,wall}} \left(\frac{K_{wall}}{\Delta z}\left(\frac{\partial T_w}{\partial z}|_{i+1/2} - \frac{\partial T_w}{\partial z}|_{i-1/2}\right)
    # + \frac{2 r_{in} \ h_{bed}}{r_{out}^2-r_{in}^2} \left(T-T_w \right) - \frac{2 r_{out} \ h_{wall}}{r_{out}^2-r_{in}^2} \left(T_w-T_a \right)\right)"""

    # =========================================================================
    # COMPONENT MASS BALANCES
    # =========================================================================

    def calculate_component_balance(yi, yi_walls, yi_all, dnidt=None):
        """Helper function to calculate component mass balance"""
        pressure_effect = -yi / P * dPdt
        temperature_effect = yi / T * dTdt

        adsorption_effect = 0
        if dnidt is not None:
            adsorption_effect = (
                - 1
                / config.adsorbent.total_voidage
                * config.fluid.R
                * T
                / P
                * dnidt
            )

        convection_effect = (
            -1
            / config.adsorbent.total_voidage
            * T
            / P
            * 1
            / column_grid["deltaZ"][1:-1]
            * (
                P_walls[1 : num_cells + 1]
                * v_walls[1 : num_cells + 1]
                * yi_walls[1 : num_cells + 1]
                / T_walls[1 : num_cells + 1]
                - P_walls[:num_cells]
                * v_walls[:num_cells]
                * yi_walls[:num_cells]
                / T_walls[:num_cells]
            )
        )

        dyidz = (yi_all[1 : num_cells + 2] - yi_all[0 : num_cells + 1]) / (
            column_grid["xCentres"][1 : num_cells + 2]
            - column_grid["xCentres"][0 : num_cells + 1]
        )

        dispersion_effect = (
            + 1
            / config.adsorbent.total_voidage
            * D_l
            * T
            / P
            * 1
            / column_grid["deltaZ"][1:-1]
            * (
                P_walls[1 : num_cells + 1]
                / T_walls[1 : num_cells + 1]
                * dyidz[1 : num_cells + 1]
                - P_walls[0:num_cells] / T_walls[0:num_cells] * dyidz[0:num_cells]
            )
        )

        return (
            pressure_effect
            + temperature_effect
            + adsorption_effect
            + convection_effect
            + dispersion_effect
        )

    # Component 1 (adsorbing)
    dy1dt = calculate_component_balance(y1, y1_walls, y1_all, dn1dt)
    # """ \dfrac{\partial y_1}{\partial t} = \dfrac{y_1}{P}\dfrac{\partial P}{\partial t} + \dfrac{y_1}{T}\dfrac{\partial T}{\partial t}
    # - \dfrac{(1-\varepsilon)}{\varepsilon}\frac{RT}{P}\frac{\partial q_1}{\partial t} - \frac{1}{\varepsilon}\frac{T}{P}\frac{1}{\Delta z}
    # \left( \frac{P_{i+1/2} \ y_{1,i+1/2} \ v_{i+1/2}}{T_{i+1/2}}-\frac{P_{i-1/2} \ y_{1,i-1/2} \ v_{i-1/2}}{T_{i-1/2}} \right) \\
    # + D_l \  \frac{T}{P} \ \frac{1}{\Delta z} \left( \frac{P_{i+1/2}}{T_{i+1/2}} \ \frac{y_{1,i+1} - y_{1,i}}{\Delta z} - \frac{P_{i-1/2}}{T_{i-1/2}} \ \frac{y_{1,i} - y_{1,i-1}}{\Delta z}\right)"""

# Non-discretized version
#     \dfrac{\partial y_i}{\partial t} = \dfrac{y_i}{P}\dfrac{\partial P}{\partial t} + \dfrac{y_i}{T}\dfrac{\partial T}{\partial t}
# - \dfrac{(1-\varepsilon_b)}{\varepsilon_t}\frac{RT}{P}\frac{\partial q_i}{\partial t} \\- \frac{\varepsilon_b}{\varepsilon_t}\frac{T}{P}\frac{\partial}{\partial z}
#  \left( \frac{P \ y_i \ v}{T} \right)
#  + D_l \frac{\varepsilon_b}{\varepsilon_t}\  \frac{T}{P} \ \frac{\partial}{\partial z} \left( \frac{P}{T} \ \frac{\partial y_i}{\partial z}\right)

    # Component 2 (adsorbing)
    dy2dt = calculate_component_balance(y2, y2_walls, y2_all, dn2dt)
    # """\dfrac{\partial y_2}{\partial t} = \dfrac{y_2}{P}\dfrac{\partial P}{\partial t} + \dfrac{y_2}{T}\dfrac{\partial T}{\partial t} - \dfrac{(1-\varepsilon)}{\varepsilon}\frac{RT}{P}\frac{\partial q_2}{\partial t}
    # - \frac{1}{\varepsilon}\frac{T}{P}\frac{1}{\Delta z} \left( \frac{P_{i+1/2} \ y_{2,i+1/2} \ v_{i+1/2}}{T_{i+1/2}}-\frac{P_{i-1/2} \ y_{2,i-1/2} \ v_{i-1/2}}{T_{i-1/2}} \right)
    # \\  + D_l \  \frac{T}{P} \ \frac{1}{\Delta z} \left( \frac{P_{i+1/2}}{T_{i+1/2}} \ \frac{y_{2,i+1} - y_{2,i}}{\Delta z} - \frac{P_{i-1/2}}{T_{i-1/2}} \ \frac{y_{2,i} - y_{2,i-1}}{\Delta z}\right)"""

    # Component 3 (non-adsorbing)
    dy3dt = calculate_component_balance(y3, y3_walls, y3_all, dn3dt)
    # """ \dfrac{\partial y_3}{\partial t} = \dfrac{y_3}{P}\dfrac{\partial P}{\partial t} + \dfrac{y_3}{T}\dfrac{\partial T}{\partial t}
    # - \frac{1}{\varepsilon}\frac{T}{P}\frac{1}{\Delta z} \left( \frac{P_{i+1/2} \ y_{3,i+1/2} \ v_{i+1/2}}{T_{i+1/2}}-\frac{P_{i-1/2} \ y_{3,i-1/2} \ v_{i-1/2}}{T_{i-1/2}} \right)
    # \\ + D_l \  \frac{T}{P} \ \frac{1}{\Delta z} \left( \frac{P_{i+1/2}}{T_{i+1/2}} \ \frac{y_{3,i+1} - y_{3,i}}{\Delta z} - \frac{P_{i-1/2}}{T_{i-1/2}} \ \frac{y_{3,i} - y_{3,i-1}}{\Delta z}\right)"""

    # =========================================================================
    # MASS AND ENERGY BALANCE TRACKING
    # =========================================================================

    # Inlet and outlet calculations for mass balance error
    dF1dt = (
        config.adsorbent.bed_voidage
        * config.geometry.inner_bed_radius ** 2
        * np.pi
        / (config.fluid.R * T_walls[0])
        * v_walls[0]
        * P_walls[0]
        * y1_walls[0]
    )
    dF2dt = (
        config.adsorbent.bed_voidage
        * config.geometry.inner_bed_radius ** 2
        * np.pi
        / (config.fluid.R * T_walls[0])
        * v_walls[0]
        * P_walls[0]
        * y2_walls[0]
    )
    dF3dt = (
        config.adsorbent.bed_voidage
        * config.geometry.inner_bed_radius ** 2
        * np.pi
        / (config.fluid.R * T_walls[0])
        * v_walls[0]
        * P_walls[0]
        * y3_walls[0]
    )
    dF4dt = (
        config.adsorbent.bed_voidage
        * config.geometry.inner_bed_radius ** 2
        * np.pi
        / (config.fluid.R * T_walls[0])
        * v_walls[0]
        * P_walls[0]
        * (1 - y1_walls[0] - y2_walls[0] - y3_walls[0])
    )

    dF5dt = (
        config.adsorbent.bed_voidage
        * config.geometry.inner_bed_radius ** 2
        * np.pi
        / (config.fluid.R * T_walls[-1])
        * v_walls[-1]
        * P_walls[-1]
        * y1_walls[-1]
    )
    dF6dt = (
        config.adsorbent.bed_voidage
        * config.geometry.inner_bed_radius ** 2
        * np.pi
        / (config.fluid.R * T_walls[-1])
        * v_walls[-1]
        * P_walls[-1]
        * y2_walls[-1]
    )
    dF7dt = (
        config.adsorbent.bed_voidage
        * config.geometry.inner_bed_radius ** 2
        * np.pi
        / (config.fluid.R * T_walls[-1])
        * v_walls[-1]
        * P_walls[-1]
        * y3_walls[-1]
    )
    dF8dt = (
        config.adsorbent.bed_voidage
        * config.geometry.inner_bed_radius ** 2
        * np.pi
        / (config.fluid.R * T_walls[-1])
        * v_walls[-1]
        * P_walls[-1]
        * (1 - y1_walls[-1] - y2_walls[-1] - y3_walls[-1])
    )

    dFdt = np.array([dF1dt, dF2dt, dF3dt, dF4dt, dF5dt, dF6dt, dF7dt, dF8dt])

    # Inlet and outlet calculations for energy balance error
    dE1dt = (
        config.adsorbent.bed_voidage
        * config.geometry.inner_bed_radius ** 2
        * np.pi
        * Cp_g[0]
        * v_walls[0]
        * T_walls[0]
        * P_walls[0]
        / (config.fluid.R * T_walls[0])
    )
    dE2dt = (
        config.adsorbent.bed_voidage
        * config.geometry.inner_bed_radius ** 2
        * np.pi
        * Cp_g[-1]
        * v_walls[-1]
        * T_walls[-1]
        * P_walls[-1]
        / (config.fluid.R * T_walls[-1])
    )
    dE3dt = np.sum(
        2
        * np.pi
        * config.geometry.outer_bed_radius
        * h_wall
        * (Tw - config.operation.ambient_temperature)
        * column_grid["deltaZ"][1:-1]
    )
    dE4dt = (
        -1
        * np.sum(
            2
            / config.geometry.inner_bed_radius
            * h_bed
            * (T - Tw)
            * column_grid["deltaZ"][1:-1]
        )
        * np.pi
        * config.geometry.inner_bed_radius ** 2
    )  # Thermal energy (J/s)
    dE5dt = -1 * (
        (dF5dt + dF7dt + dF8dt)
        / config.efficiency.compressor_efficiency
        * (config.fluid.k / (1 - config.fluid.k))
        * config.fluid.R
        * T_walls[-1]
        * (
            (config.operation.ambient_pressure / P_walls[-1])
            ** ((config.fluid.k - 1) / (config.fluid.k))
            - 1
        )
    )  # Compressor work (J/s)
    dE6dt = -1 * (
        (dF1dt + dF2dt + dF3dt + dF4dt)
        / config.efficiency.fan_efficiency
        * (config.fluid.k / (1 - config.fluid.k))
        * config.fluid.R
        * T_walls[0]
        * (
            (P_walls[0] / config.operation.ambient_pressure)
            ** ((config.fluid.k - 1) / (config.fluid.k))
            - 1
        )
    )  # Fan work (J/s)
    dE7dt = (
        config.fluid.MW_H2O
        / 1000
        * (
            dF2dt
            * config.fluid.Cp_water
            * (
                func.H2O_boiling_point(P_walls[0])
                - config.operation.ambient_temperature
            )
            + dF2dt
            * config.fluid.vaporization_energy
            * dF2dt
            * config.fluid.Cp_steam
            * (config.steam.temperature - func.H2O_boiling_point(P_walls[0]))
        )
    )
    # Sensible energy to heat water to boiling point (J/s)

    dEdt = np.array([dE1dt, dE2dt, dE3dt, dE4dt, dE5dt, dE6dt, dE7dt])
    # Combine derivatives into a single vector
    if column_direction == "forwards":
        derivatives = np.concatenate(
            [
                dPdt / config.reference.P_ref,
                dTdt / config.reference.T_ref,
                dTwdt / config.reference.T_ref,
                dy1dt,
                dy2dt,
                dy3dt,
                dn1dt / config.reference.n_ref,
                dn2dt / config.reference.n_ref,
                dn3dt / config.reference.n_ref,
                dFdt,
                dEdt,
            ]
        )
    elif column_direction == "reverse":
        P = np.flip(P)
        T = np.flip(T)
        Tw = np.flip(Tw)
        y1 = np.flip(y1)
        y2 = np.flip(y2)
        y3 = np.flip(y3)
        n1 = np.flip(n1)
        n2 = np.flip(n2)
        n3 = np.flip(n3)
        F = np.flip(F)
        E = np.flip(E)
        derivatives = np.concatenate(
            [
                dPdt / config.reference.P_ref,
                dTdt / config.reference.T_ref,
                dTwdt / config.reference.T_ref,
                dy1dt,
                dy2dt,
                dy3dt,
                dn1dt / config.reference.n_ref,
                dn2dt / config.reference.n_ref,
                dn3dt / config.reference.n_ref,
                dFdt,
                dEdt,
            ]
        )
    # Combine derivatives into a single vector
    return derivatives


def final_wall_values(
    column_grid, config, left_values, right_values, output_matrix
):
    """
    Calculate wall values for each time step in the adsorption column simulation.

    Returns:
        Arrays with shape (n_walls, n_timesteps) for each wall variable.
    """
    import numpy as np

    # Initialize lists to accumulate time series
    P_walls_ = []
    T_walls_ = []
    Tw_walls_ = []
    y1_walls_ = []
    y2_walls_ = []
    y3_walls_ = []
    v_walls_ = []

    num_cells = column_grid["num_cells"]
    num_timesteps = output_matrix.t.shape[0]

    for t in range(num_timesteps):
        # Get state vector at timestep t

        # Unpack variables from the state
        P, T, Tw, y1, y2, y3, n1, n2, n3, F, E = data_prep(
            output_matrix.y[:, t], num_cells, config
        )

        # Boundary conditions
        (
            P_left,
            T_left,
            Tw_left,
            y1_left,
            y2_left,
            y3_left,
            v_left,
            dPdz_left,
            dTwdz_left,
            dTdz_left,
        ) = left_boundary_conditions(
            P, T, Tw, y1, y2, y3, column_grid, config, left_values, t
        )

        (
            P_right,
            T_right,
            Tw_right,
            y1_right,
            y2_right,
            y3_right,
            v_right,
            dPdz_right,
            dTwdz_right,
            dTdz_right,
        ) = right_boundary_conditions(
            P, T, Tw, y1, y2, y3, column_grid, config, right_values, t
        )

        # Ghost cells
        P_all, T_all, Tw_all, y1_all, y2_all, y3_all = ghost_cell_calculations(
            P,
            T,
            Tw,
            y1,
            y2,
            y3,
            P_left,
            P_right,
            T_left,
            T_right,
            Tw_left,
            Tw_right,
            y1_left,
            y1_right,
            y2_left,
            y2_right,
            y3_left,
            y3_right,
            column_grid,
        )

        # Wall values for this timestep
        (
            P_walls,
            T_walls,
            Tw_walls,
            y1_walls,
            y2_walls,
            y3_walls,
            v_walls,
            dTdz_walls,
            dTwdz_walls,
        ) = calculate_internal_wall_values(
            P_all,
            T_all,
            Tw_all,
            y1_all,
            y2_all,
            y3_all,
            P_left,
            P_right,
            T_left,
            T_right,
            Tw_left,
            Tw_right,
            y1_left,
            y1_right,
            y2_left,
            y2_right,
            y3_left,
            y3_right,
            v_left,
            v_right,
            dPdz_left,
            dPdz_right,
            dTwdz_left,
            dTwdz_right,
            dTdz_left,
            dTdz_right,
        config,
            column_grid,
            left_values,
        )

        # Append current timestep wall values
        P_walls_.append(P_walls)
        T_walls_.append(T_walls)
        Tw_walls_.append(Tw_walls)
        y1_walls_.append(y1_walls)
        y2_walls_.append(y2_walls)
        y3_walls_.append(y3_walls)
        v_walls_.append(v_walls)

    # Convert to arrays of shape (n_walls, n_timesteps)
    return (
        np.array(P_walls_).T,
        np.array(T_walls_).T,
        np.array(Tw_walls_).T,
        np.array(y1_walls_).T,
        np.array(y2_walls_).T,
        np.array(y3_walls_).T,
        np.array(v_walls_).T,
    )
