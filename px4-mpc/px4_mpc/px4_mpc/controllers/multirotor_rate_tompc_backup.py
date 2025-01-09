#!/usr/bin/env python3

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
import numpy as np
import scipy.linalg
import casadi as cs


class MultirotorRateTOMPC():
    def __init__(self, model, reference_trajectory):
        """
        model                : An instance of your MultirotorRateModel (with get_acados_model()).
        reference_trajectory : np.array of shape (N, 16), each row containing
                               [p_ref(3), T_ref(3), N_ref(3), B_ref(3), q_ref(4)].
        """
        self.model = model
        self.reference_trajectory = reference_trajectory
        self.Tf = 1.0  # Time horizon
        self.N = 30    # Number of discretization steps

        # Initial state placeholder: [x, y, z, vx, vy, vz, qw, qx, qy, qz]
        self.x0 = np.array([0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0,
                            1.0, 0.0, 0.0, 0.0])
        
        self.ocp_solver, self.integrator = self.setup(self.x0, self.N, self.Tf)

    def setup(self, x0, N_horizon, Tf):
        # Create OCP object
        ocp = AcadosOcp()

        # Get the AcadosModel from the rate model
        acados_model = self.model.get_acados_model()

        # Assign model properties to the OCP
        ocp.model = acados_model
        ocp.dims.N = N_horizon

        # ---------------------------------------------------------------------
        # 1) Define the 3D reference parameters as a single parameter vector
        #    p_ref(3), T_ref(3), N_ref(3), B_ref(3), q_ref(4) = 16 total
        # ---------------------------------------------------------------------
        p_ref = cs.MX.sym("p_ref", 3)     # reference position
        T_ref = cs.MX.sym("T_ref", 3)     # Frenet tangent
        N_ref = cs.MX.sym("N_ref", 3)     # Frenet normal
        B_ref = cs.MX.sym("B_ref", 3)     # Frenet binormal
        qw_ref = cs.MX.sym("qw_ref")
        qx_ref = cs.MX.sym("qx_ref")
        qy_ref = cs.MX.sym("qy_ref")
        qz_ref = cs.MX.sym("qz_ref")

        # Combine into a single parameter vector of length 16
        ocp.model.p = cs.vertcat(
            p_ref,     # 3
            T_ref,     # 3
            N_ref,     # 3
            B_ref,     # 3
            qw_ref, qx_ref, qy_ref, qz_ref  # 4
        )
        ocp.parameter_values = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # default initialization

        # ---------------------------------------------------------------------
        # 2) Extract your actual state and define 3D MPCC errors
        # ---------------------------------------------------------------------
        # States: x, y, z, vx, vy, vz, qw, qx, qy, qz
        x_pos = acados_model.x[0]   # x
        y_pos = acados_model.x[1]   # y
        z_pos = acados_model.x[2]   # z
        vx    = acados_model.x[3]
        vy    = acados_model.x[4]
        vz    = acados_model.x[5]
        qw    = acados_model.x[6]
        qx    = acados_model.x[7]
        qy    = acados_model.x[8]
        qz    = acados_model.x[9]

        roll_angle = cs.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
        pitch_angle = cs.asin(2 * (qw * qy - qz * qx))
        psi_angle = cs.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))


        # Position error
        p = cs.vertcat(x_pos, y_pos, z_pos)       # current position
        p_ref_expr = p_ref                        # reference position
        dp = p - p_ref_expr

        # Frenet-Serret vectors
        # T, N, B are each (3x1) here
        # dot() is used to project dp onto T, N, B
        e_l = cs.dot(dp, T_ref)  # "lag" error along tangent
        e_c = cs.dot(dp, N_ref)  # "contour" error in normal direction
        e_b = cs.dot(dp, B_ref)  # "binormal" or out-of-plane error

        # ---------------------------------------------------------------------
        # 3) Quaternion orientation error
        # ---------------------------------------------------------------------
        # Minimal approach: 1 - dot(q, q_ref).
        # dot(q, q_ref) = qw*qw_ref + qx*qx_ref + qy*qy_ref + qz*qz_ref
        q_ref_dot = qw * qw_ref + qx * qx_ref + qy * qy_ref + qz * qz_ref
        e_quat = 1.0 - q_ref_dot

        # ---------------------------------------------------------------------
        # 4) Define cost expressions (error states + velocity + controls)
        # ---------------------------------------------------------------------
        # Controls: F, wx, wy, wz
        # This is the same structure you used before, just new error definitions
        ocp.cost.cost_type   = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"

        # full-stage cost
        ocp.model.cost_y_expr = cs.vertcat(
            e_l, e_c, e_b,        # 3
            e_quat,               # 1
            vx, vy, vz,           # 3
            acados_model.u        # 4 controls
        )  # total = 3 + 1 + 3 + 4 = 11

        # terminal cost
        ocp.model.cost_y_expr_e = cs.vertcat(
            e_l, e_c, e_b, e_quat
        )  # 4 terms

        # ---------------------------------------------------------------------
        # 5) Weight matrices
        # ---------------------------------------------------------------------
        # Example weights
        # e_l, e_c, e_b, e_quat, vx, vy, vz, (F, wx, wy, wz)
        W_stage = scipy.linalg.block_diag(
            150.0,   # weight on e_l (lag error) (higher stricter)
            400.0,  # weight on e_c (contour error) (higher stricter)
            100.0,  # weight on e_b (height error) (higher stricter)
            2.0,   # weight on e_quat (orientation error) (higher stricter)
            np.diag([7.0, 7.0, 100.0]),  # velocity (higher smoother)
            np.diag([3.0, 3.0, 3.0, 3.0])  # controls (higher less control input)
        )
        # dimension should match the total 11 elements in cost_y_expr
        ocp.cost.W = W_stage

        # Terminal weights: e_l, e_c, e_b, e_quat
        W_terminal = np.diag([10.0, 100.0, 10.0, 20.0])
        ocp.cost.W_e = W_terminal

        # Reference vectors: dimension must match cost_y_expr
        ocp.cost.yref  = np.zeros(11)  # 3 + 1 + 3 + 4
        ocp.cost.yref_e = np.zeros(4)  # 4

        # ---------------------------------------------------------------------
        # 6) Constraints
        # ---------------------------------------------------------------------
        # Example bounds on [F, wx, wy, wz]
        ocp.constraints.lbu = np.array([
            0.0, 
            -self.model.max_rate, 
            -self.model.max_rate, 
            -0.5 * self.model.max_rate
        ])
        ocp.constraints.ubu = np.array([
            self.model.max_thrust, 
            self.model.max_rate, 
            self.model.max_rate, 
            0.5 * self.model.max_rate
        ])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])

        # ocp.constraints.lbx = np.array([-0.5])
        # ocp.constraints.ubx = np.array([2.0])
        # ocp.constraints.idxbx = np.array([5])


        # initial state
        ocp.constraints.x0 = x0

        # Quaternion normalization constraint
        quat_norm_constraint = qw**2 + qx**2 + qy**2 + qz**2 - 1

        # Roll, Pitch, and Yaw from quaternion
        roll = cs.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
        pitch = cs.asin(2 * (qw * qy - qz * qx))
        yaw = cs.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))

        # Bounds on roll, pitch, and yaw
        roll_max = np.pi / 12   # 30 degrees
        pitch_max = np.pi / 12  # 45 degrees
        # yaw_max = np.pi        # 180 degrees

        # Roll, pitch, and yaw constraints
        roll_constraints = cs.vertcat(
            roll - roll_max,  # roll <= roll_max
            -roll - roll_max  # roll >= -roll_max
        )
        pitch_constraints = cs.vertcat(
            pitch - pitch_max,  # pitch <= pitch_max
            -pitch - pitch_max  # pitch >= -pitch_max
        )
        # yaw_constraints = cs.vertcat(
        #     yaw - yaw_max,  # yaw <= yaw_max
        #     -yaw - yaw_max  # yaw >= -yaw_max
        # )

        # # Combine all constraints
        ocp.model.con_h = cs.vertcat(
            quat_norm_constraint,  # Quaternion normalization
            roll_constraints,      # Roll constraints
            pitch_constraints,     # Pitch constraints
        #     # yaw_constraints         # Yaw constraints
        )

        # # Define bounds for combined constraints
        # # num_constraints = ocp.model.con_h.size1()
        # # ocp.constraints.lh = -1e-8 * np.ones(num_constraints)  # Allow small tolerance for numerical errors
        # # ocp.constraints.uh = np.zeros(num_constraints)         # Upper bounds as zeros for <= constraints



        # ---------------------------------------------------------------------
        # 7) Solver Options
        # ---------------------------------------------------------------------
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 3
        ocp.solver_options.qp_solver_cond_N = N_horizon
        ocp.solver_options.tf = Tf

        # Create the ACADOS solvers
        ocp_solver       = AcadosOcpSolver(ocp, json_file="acados_ocp.json")
        acados_integrator = AcadosSimSolver(ocp, json_file="acados_ocp.json")

        return ocp_solver, acados_integrator

    def solve(self, x0, ref_traj, verbose=False):
        """
        x0       : Current state (10, ) or (10,1)
        ref_traj : Array of shape (N,16) containing 
                   [pRef(3), TRef(3), NRef(3), BRef(3), qW_ref, qX_ref, qY_ref, qZ_ref]
        """
        ocp_solver = self.ocp_solver

        # 1) Set the initial state
        ocp_solver.set(0, "lbx", x0)
        ocp_solver.set(0, "ubx", x0)

        # 2) Update parameters for each shooting node
        #    shape(ref_traj[i]) = (16,)
        for i in range(self.N):
            ocp_solver.set(i, "p", ref_traj[i])

        # 3) Solve
        status = ocp_solver.solve()
        if verbose:
            ocp_solver.print_statistics()

        if status != 0:
            raise Exception(f"acados returned status {status}.")

        # 4) Retrieve solution
        N = self.N
        nx = self.model.get_acados_model().x.size1()
        nu = self.model.get_acados_model().u.size1()

        simX = np.zeros((N + 1, nx))
        simU = np.zeros((N, nu))

        for i in range(N):
            simX[i, :] = ocp_solver.get(i, "x")
            simU[i, :] = ocp_solver.get(i, "u")
        simX[N, :] = ocp_solver.get(N, "x")

        return simU, simX


# --------------------------------------------------------------------------
# Example usage (pseudo-code):
#
# model = MultirotorRateModel()
# # Suppose you have a function get_3d_reference(N) that returns an array
# # of shape (N, 16): [px, py, pz, Tx, Ty, Tz, Nx, Ny, Nz, Bx, By, Bz, qw, qx, qy, qz]
# reference_trajectory = get_3d_reference(15)
#
# # Create the 3D MPC object
# mpc_3d = MultirotorRate3DTOMPC(model, reference_trajectory)
#
# # Solve with some initial state x0
# x0_current = np.array([...])  # your real quadrotor state
# u_pred, x_pred = mpc_3d.solve(x0_current, reference_trajectory, verbose=True)
# --------------------------------------------------------------------------
