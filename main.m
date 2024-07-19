clc;
clear;
close all;

check_acados_requirements()

%% discretization
N = 50;
T = 5;

% Initial conditions
Xi_0 = [0,0]; % Initial position
Vi_0 = [-4,6];  % Initial velocity
x0 = horzcat(Xi_0, Vi_0);
u0 = [0,0]; % Initial acc command

nlp_solver = 'sqp'; % sqp, sqp_rti

% integrator type
sim_method = 'erk'; % erk = explicit Runge Kutta, irk = implicit Runge Kutta, irk_gnsf

%% model dynamics
ocp_model = test_model;
nx = ocp_model.model_struct.nx;
nu = ocp_model.model_struct.nu;

max_velocity_squared_xy = 100;
max_acc_squared_xy = 9;

ocp_model.model_struct.constr_lh = [-1,-1]'; % Not 0 because then 0 initial conditions are problematic
ocp_model.model_struct.constr_uh = [max_velocity_squared_xy,max_acc_squared_xy]';

ocp_model.model_struct.constr_lh_0 = [-1,-1]'; % Not 0 because then 0 initial conditions are problematic
ocp_model.model_struct.constr_uh_0 = [max_velocity_squared_xy,max_acc_squared_xy]';

ocp_model.set('constr_x0', x0);
% ocp_model.set('cost_type', 'ext_cost')
% ocp_model.set('cost_type_0', 'ext_cost')
% ocp_model.set('cost_type_e', 'ext_cost')

% dynamics
ocp_model.set('dyn_type', 'explicit');
ocp_model.set('dyn_expr_f', ocp_model.model_struct.expr_f_expl);

ocp_model.model_struct.T = T;

%% acados ocp set opts
ocp_opts = acados_ocp_opts();
ocp_opts.set('sim_method', sim_method);
ocp_opts.set('param_scheme_N', N);
ocp_opts.set('nlp_solver', nlp_solver);
ocp_opts.set('nlp_solver_max_iter', 100);
ocp_opts.set('regularize_method', 'mirror');
ocp_opts.set('nlp_solver_exact_hessian', 'true');

disp(ocp_model.model_struct.cost_expr_ext_cost)

ocp = acados_ocp(ocp_model, ocp_opts);
x_traj_init = repmat(x0', 1,N+1);
u_traj_init = repmat(u0', 1,N);

%% call ocp solver

% set trajectory initialization
ocp.set('init_x', x_traj_init);
ocp.set('init_u', u_traj_init);

Xi(:,1) = Xi_0;
x(:,1) = x0';

ocp.solve();
ocp.print('stat')

switch ocp.get('status')
    case 0
        ocp.print('stat')
    case 1
        error('Failed: Detected NAN');
    case 2
        error('Failed: Maximum number of iterations reached');
    case 3
        error('Failed: Minimum step size in QP solver reached');
    case 4
        error('Failed: QP solver failed');
end

% get solution
u_result = ocp.get('u');
x_result = ocp.get('x');

acc_norm= zeros(1,size(u_result,2));

position(:,1) = Xi_0;
Vi(:,1) = Vi_0;
Vi_norm(:,1) = norm(Vi_0);

for j = 1:1:size(u_result,2)

    acc_norm(j) = norm(u_result(:,j));
    position(1:2,j) = x_result(1:2,j);
    Vi(1:2,j) = x_result(3:4,j);
    Vi_norm(j) = norm(Vi(:,j));

end

%% Plots
figure;
plot(position(1,:),position(2,:));
title('Position XY');
legend('Inteceptor Position');
grid on;
xlabel('X [m]');
ylabel('Y [m]');

figure;
dt = T/N;
time = dt:dt:T;
plot(time, acc_norm);
title('Acceleration XY Magnitude');
ylabel('Acceleration XY [m/sec^2]')
xlabel('Time [sec]')
grid on;

figure;
plot(time, Vi_norm);
title('Velocity XY Magnitude');
ylabel('Velocity XY [m/sec]')
xlabel('Time [sec]')
grid on;


function model = test_model()

import casadi.*

%% system dimensions
nx = 4;
nu = 2;

%% named symbolic variables
pos = SX.sym('pos',2);
vel = SX.sym('vel',2);
acc = SX.sym('acc',2);

%% (unnamed) symbolic variables
sym_x = vertcat(pos,vel);

sym_xdot = SX.sym('xdot', nx, 1);
sym_u = acc;

%% dynamics
expr_f_expl = vertcat(vel, acc);
expr_f_impl = expr_f_expl - sym_xdot;

%% constraints
expr_h = [vel(1)*vel(1)+vel(2)*vel(2) , acc(1)*acc(1)+acc(2)*acc(2)]'; % Magnitude of vel and acc squared

%% cost
W_x = diag([100, 100,0,0]); % Weight is only on position, no weight on velocity
W_u = diag([1, 1]);

P_des = [100, -50]'; % Desired Position
V_des = [0, 0]'; % Desired velocity, but no weight is applied to it

X_des = vertcat(P_des, V_des);

expr_ext_cost_e = (X_des - sym_x)'* W_x * (X_des - sym_x);
expr_ext_cost = expr_ext_cost_e + sym_u' * W_u * sym_u;

W = blkdiag(W_x, W_u);

model = acados_ocp_model();

%% populate structure
model.model_struct.name = 'Test';
model.model_struct.nx = nx;
model.model_struct.nu = nu;
model.model_struct.sym_x = sym_x;
model.model_struct.sym_xdot = sym_xdot;
model.model_struct.sym_u = sym_u;
model.model_struct.expr_f_expl = expr_f_expl;
model.model_struct.expr_f_impl = expr_f_impl;
model.model_struct.constr_expr_h = expr_h;
model.model_struct.constr_expr_h_0 = expr_h;
% model.model_struct.constr_expr_h_e = vel(1)*vel(1)+vel(2)*vel(2);
model.model_struct.cost_expr_ext_cost_0 = expr_ext_cost;

model.model_struct.constr_type = 'auto';

model.model_struct.cost_expr_ext_cost = expr_ext_cost;
model.model_struct.cost_expr_ext_cost_e = expr_ext_cost_e;
model.model_struct.cost_W= W;
model.model_struct.cost_W_e= W_x;
end