from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel, latexify_plot
import numpy as np
from matplotlib import pyplot as plt
import casadi as cs

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

def main():

    # The flag denotes, if the problem should be transformed into a feasibility
    # problem, or if the unconstrained OCP should be solved.
    SOLVE_FEASIBILITY_PROBLEM = False

    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = export_double_integrator_model()
    ocp.model = model

    Tf = 5.0 # It is a time optimal problem with time scaling
    nx = model.x.rows()
    nu = model.u.rows()
    N = 50

    # set dimensions
    ocp.dims.N = N

    Xi_0 = [0,0] # Initial position
    Vi_0 = [-4,6] #  Initial velocity
    x0 = np.array(Xi_0 + Vi_0)

    # the 'EXTERNAL' cost type can be used to define general cost terms
    # NOTE: This leads to additional (exact) hessian contributions when using GAUSS_NEWTON hessian.
    ###########################################################################
    # Define cost
    ###########################################################################
    W_x = cs.diag(cs.vertcat(100, 100,0,0))
    W_u = cs.diag(cs.vertcat(1, 1))

    P_des = cs.vertcat(100, -50) # Desired Position
    V_des = cs.vertcat(0, 0) # Desired velocity, but no weight is applied to it
    X_des = cs.vertcat(P_des, V_des)

    cost_x = (X_des - model.x).T @ W_x @ (X_des - model.x)

    ocp.cost.cost_type = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost = cost_x + model.u.T @ W_u @ model.u
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost_e = cost_x

    ###########################################################################
    # Define constraints
    ###########################################################################

    # Initial conditions
    ocp.constraints.lbx_0 = x0
    ocp.constraints.ubx_0 = x0
    ocp.constraints.idxbx_0 = np.arange(nx)

    # Nonlinear Constraints
    max_velocity_squared_xy = 100
    max_acc_squared_xy = 9
    max_speed_and_velocity = np.array([max_velocity_squared_xy, max_acc_squared_xy])
    lower_bound = np.array([-1e8, -1e8])
    # lower_bound = np.arravy([0, 0]) # this one gives problems for QP solvers!
    squared_velocity_and_constraints = cs.vertcat(cs.sumsqr(ocp.model.x[:2]),
                                                  cs.sumsqr(ocp.model.x[2:]))
    # Over path
    ocp.model.con_h_expr = squared_velocity_and_constraints
    ocp.constraints.uh = max_speed_and_velocity
    ocp.constraints.lh = lower_bound

    # terminal constraints
    ocp.model.con_h_expr_e = squared_velocity_and_constraints
    ocp.constraints.uh_e = max_speed_and_velocity
    ocp.constraints.lh_e = lower_bound

    ###########################################################################
    # set solver options
    ###########################################################################
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    # ocp.solver_options.qp_solver_cond_N = N
    # ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.integrator_type = 'ERK'
    # ocp.solver_options.sim_method_num_steps = M
    ocp.solver_options.print_level = 1

    # DDP options
    ocp.solver_options.nlp_solver_max_iter = 100
    ocp.solver_options.nlp_solver_type = 'SQP'
    # ocp.solver_options.globalization = 'FUNNEL_L1PEN_LINESEARCH'
    ocp.solver_options.with_adaptive_levenberg_marquardt = False
    ocp.solver_options.regularize_method = 'MIRROR'

    # set prediction horizon
    ocp.solver_options.tf = Tf

    if SOLVE_FEASIBILITY_PROBLEM:
        ocp.translate_to_feasibility_problem()

    ocp_solver = AcadosOcpSolver(ocp, json_file = 'simple_double_integrator.json')

    # for i in range(N):
    #     ocp_solver.cost_set(i, "scaling", 1.0)
    sol_X = np.zeros((N+1, nx))
    sol_U = np.zeros((N, nu))

    for i in range(N):
        ocp_solver.set(i, "x", x0)
        # ocp_solver.set(i, "u", U_init[:,i])
    ocp_solver.set(N, "x", x0)

    # Solve the problem
    status = ocp_solver.solve()

    # iter = ocp_solver.get_stats('nlp_iter')
    # assert iter <= 14, "DDP Solver should converge within 14 iterations!"

    # if status != 0:
    #     raise Exception(f'acados returned status {status}.')

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