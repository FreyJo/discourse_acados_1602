from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel, latexify_plot
import numpy as np
from matplotlib import pyplot as plt
import casadi as cs
from scipy.linalg import block_diag

def export_double_integrator_model():

    model_name = 'double_integrator_2d'
    # system dimensions
    nx = 4

    ## named symbolic variables
    # define states
    pos = cs.SX.sym('pos',2)
    vel = cs.SX.sym('vel',2)
    states = cs.vertcat(pos, vel)

    # define controls
    acc = cs.SX.sym('acc',2)
    controls = cs.vertcat(acc)

    # define dynamics expression
    states_dot = cs.SX.sym('xdot', nx, 1)

    ## dynamics
    f_expl = cs.vertcat(vel, acc)
    f_impl = states_dot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = states
    model.xdot = states_dot
    model.u = controls
    model.name = model_name

    return model

def main(modification=9):
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = export_double_integrator_model()
    ocp.model = model

    Tf = 5.0
    nx = model.x.rows()
    nu = model.u.rows()
    N = 50

    # set dimensions
    ocp.dims.N = N

    Xi_0 = [0,0] # Initial position
    Vi_0 = [-4,6] #  Initial velocity
    u_init = np.array([-0.0, -0.0])

    globalization = 'FIXED_STEP'
    qp_solver_iter_max = 50
    qp_solver = 'PARTIAL_CONDENSING_HPIPM'

    if modification == 1:
        # Change initial position
        Xi_0 = [-10, 0]
        globalization = 'MERIT_BACKTRACKING'
        qp_solver_iter_max = 100
    elif modification == 2:
        # Change initial velocity
        Vi_0 = [-3,6]
    elif modification == 3:
        u_init = np.array([-1.0, -1.0])
    elif modification == 4:
        Tf = 10.0
        qp_solver = "FULL_CONDENSING_DAQP"
        qp_solver_iter_max = 200
    # other QP solver settings
    elif modification == 5:
        # fails with infeasible QP
        qp_solver_iter_max = 200
        qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    elif modification == 6:
        # works fine
        qp_solver_iter_max = 200
        qp_solver = 'FULL_CONDENSING_DAQP'
    elif modification == 7:
        # works fine
        qp_solver_iter_max = 200
        qp_solver = 'PARTIAL_CONDENSING_OSQP'
    elif modification == 8:
        # works fine
        qp_solver_iter_max = 200
        qp_solver = 'PARTIAL_CONDENSING_QPOASES'
    elif modification == 9:
        # works fine
        qp_solver_iter_max = 200
        qp_solver = 'FULL_CONDENSING_HPIPM'

    x0 = np.array(Xi_0 + Vi_0)

    # the 'EXTERNAL' cost type can be used to define general cost terms
    # NOTE: This leads to additional (exact) hessian contributions when using GAUSS_NEWTON hessian.
    ###########################################################################
    # Define cost
    ###########################################################################

    cost_type = "EXTERNAL"
    # cost_type = "LINEAR_LS"
    cost_type = "LLS_SMALL"
    W_x = cs.diag(cs.vertcat(100, 100, 0, 0))
    W_u = 1.* cs.diag(cs.vertcat(1, 1))

    P_des = cs.vertcat(100, -50) # Desired Position
    V_des = cs.vertcat(0, 0) # Desired velocity, but no weight is applied to it
    if cost_type == "EXTERNAL":
        X_des = cs.vertcat(P_des, V_des)

        cost_x = (X_des - model.x).T @ W_x @ (X_des - model.x)

        ocp.cost.cost_type = 'EXTERNAL'
        ocp.model.cost_expr_ext_cost = cost_x + model.u.T @ W_u @ model.u
        ocp.cost.cost_type_e = 'EXTERNAL'
        ocp.model.cost_expr_ext_cost_e = cost_x
    elif cost_type == "LINEAR_LS":
        ocp.cost.cost_type = 'LINEAR_LS'
        ny = nx + nu
        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[nx:, :] = np.eye(nu)
        ocp.cost.W = 2 * block_diag(W_x.full(), W_u.full())
        ocp.cost.Vx_e = np.eye(nx)
        ocp.cost.W_e = 2 * W_x.full()
        ocp.cost.yref = np.concatenate((P_des, V_des, np.zeros((nu, 1)))).flatten()
        ocp.cost.yref_e = np.concatenate((P_des, V_des)).flatten()
    elif cost_type == "LLS_SMALL":
        ocp.cost.cost_type = 'LINEAR_LS'
        ny = 4
        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:2, :2] = np.eye(2)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[2:, :] = np.eye(nu)
        ocp.cost.W = 2 * block_diag(cs.diag(cs.vertcat(100, 100)), W_u.full())
        ocp.cost.yref = np.concatenate((P_des, np.zeros((nu, 1)))).flatten()
        ocp.cost.Vx_e = ocp.cost.Vx[:2, :]
        ocp.cost.W_e = 2 * cs.diag(cs.vertcat(100, 100)).full()
        ocp.cost.yref_e = P_des.full().flatten()
    else:
        raise Exception(f'Unknown cost type {cost_type}.')

    ###########################################################################
    # Define constraints
    ###########################################################################

    # Initial conditions
    # ocp.constraints.lbx_0 = x0
    # ocp.constraints.ubx_0 = x0
    # ocp.constraints.idxbx_0 = np.arange(nx)
    ocp.constraints.x0 = x0

    # Nonlinear Constraints
    max_velocity_squared_xy = 100
    max_acc_squared_xy = 9

    constraint_formulation = "BGH"
    lh = 1e3 * np.array([-1, -1])
    uh = np.array([max_velocity_squared_xy, max_acc_squared_xy])
    if constraint_formulation == "BGH":
        squared_velocity_and_constraints = cs.vertcat(cs.sumsqr(ocp.model.x[2:]),
                                                    cs.sumsqr(ocp.model.u))
        # Over path
        ocp.model.con_h_expr = squared_velocity_and_constraints
        ocp.constraints.uh = uh
        ocp.constraints.lh = lh

        # terminal constraints
        ocp.model.con_h_expr_0 = squared_velocity_and_constraints
        ocp.constraints.uh_0 = uh
        ocp.constraints.lh_0 = lh

    else:
        ocp.model.con_r_in_phi = cs.SX.sym('con_r', 4, 1)
        ocp.model.con_phi_expr = cs.vertcat(cs.sumsqr(ocp.model.con_r_in_phi[:2]),
                                                    cs.sumsqr(ocp.model.con_r_in_phi[2:]))
        ocp.model.con_r_expr = cs.vertcat(ocp.model.x[2:], ocp.model.u)
        ocp.constraints.uphi = uh
        ocp.constraints.lphi = lh

        ocp.model.con_r_in_phi_0 = ocp.model.con_r_in_phi
        ocp.model.con_phi_expr_0 = ocp.model.con_phi_expr
        ocp.model.con_r_expr_0 = ocp.model.con_r_expr
        ocp.constraints.uphi_0 = uh
        ocp.constraints.lphi_0 = lh

    ###########################################################################
    # set solver options
    ###########################################################################
    ocp.solver_options.qp_solver = qp_solver
    # ocp.solver_options.qp_solver_cond_N = N
    # ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.integrator_type = 'ERK'
    # ocp.solver_options.sim_method_num_steps = M
    ocp.solver_options.print_level = 1
    # ocp.solver_options.nlp_solver_ext_qp_res = 1

    # DDP options
    ocp.solver_options.nlp_solver_max_iter = 100
    ocp.solver_options.qp_solver_iter_max = qp_solver_iter_max
    ocp.solver_options.nlp_solver_type = 'SQP'
    # ocp.solver_options.globalization = 'FUNNEL_L1PEN_LINESEARCH'
    ocp.solver_options.globalization = globalization
    ocp.solver_options.with_adaptive_levenberg_marquardt = False
    # ocp.solver_options.reg_epsilon = 1e-3
    ocp.solver_options.regularize_method = 'MIRROR'

    # set prediction horizon
    ocp.solver_options.tf = Tf

    ocp_solver = AcadosOcpSolver(ocp, json_file = 'simple_double_integrator.json')

    # for i in range(N):
    #     ocp_solver.cost_set(i, "scaling", 1.0)
    sol_X = np.zeros((N+1, nx))
    sol_U = np.zeros((N, nu))

    for i in range(N):
        ocp_solver.set(i, "x", x0)
        ocp_solver.set(i, "u", u_init)
    ocp_solver.set(N, "x", x0)

    # Solve the problem
    status = ocp_solver.solve()
    ocp_solver.print_statistics()

    # get solution
    for i in range(N):
        sol_X[i,:] = ocp_solver.get(i, "x")
        sol_U[i,:] = ocp_solver.get(i, "u")
    sol_X[N,:] = ocp_solver.get(N, "x")

    print("Initial state: ", sol_X[0,:])
    print("Initial control: ", sol_U[0,:])
    print("Terminal state: ", sol_X[N,:])

    # plot results
    latexify_plot()
    plt.figure()
    plt.plot(sol_X[:,0], sol_X[:,1], '-o')
    plt.title('Position trajectory')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()

    plt.figure()
    plt.title('Control trajectory')
    acc_norm = [np.linalg.norm(sol_U[i]) for i in range(N)] + [None]
    vel_norm = [np.linalg.norm(sol_X[i, 2:]) for i in range(N+1)]
    plt.plot(ocp.solver_options.shooting_nodes, acc_norm, '-o', label='acc norm')
    plt.plot(ocp.solver_options.shooting_nodes, vel_norm, '-o', label='vel norm')
    plt.grid()
    plt.legend()

    plt.show()

if __name__ == '__main__':
    main()