clear; clc; close all;

dt   = .5;                 % time step [s]
T    = 500;               % total simulation time [s]
K    = round(T/dt);       % number of time steps
N    = 40;                % MHE horizon length
sig_y1 = [0.30; 0.30];    % sensor 1 std dev [x;y]
R1 = diag(sig_y1.^2);     % sensor 1 covariance
sig_y2 = [0.05; 0.05];    % sensor 2 std dev [x;y]
R2 = diag(sig_y2.^2);     % sensor 2 covariance
P0   = diag([0.5^2, 0.5^2, (10*pi/180)^2]); % initial state covariance
Pinv = inv(P0);           % inverse prior covariance
x_big = 1e9;              % large bound on x position
y_wall = -0.01;           % wall constraint on y
sig_w = [0.05; 0.001; 2*pi/180]; % process noise std dev
Qinv  = inv(diag(sig_w.^2));     % inverse process covariance
lambda_smooth = 1e3;      % smoothing weight

A = eye(3);               % state transition matrix
B = [dt  0;               % input matrix
     0   0;
     0  dt];

H = [1 0 0;               % measurement matrix
     0 1 0];

nx = 3;                   % number of states
nu = 2;                   % number of inputs
ny = 2;                   % number of measurements
x_true = zeros(nx, K+1);  % true state trajectory
u_true = zeros(nu, K);    % true control inputs
y1 = zeros(ny, K+1);      % sensor 1 measurements
y2 = zeros(ny, K+1);      % sensor 2 measurements
x_true(:,1) = [0; 0; 0];  % initial true state
v_cmd     = 1.0;          % commanded velocity
omega_cmd = 0.0;          % commanded angular rate

rng(4);                   % worst KF case: large negative estimate early
% rng(2);                 % KF estimate goes negative
% rng(1);                 % KF never crosses the wall

% simulate the true system forward in time using the commanded inputs
for k = 1:K
    u_true(:,k) = [v_cmd; omega_cmd];           % control input
    x_true(:,k+1) = A*x_true(:,k) + B*u_true(:,k); % state update
    x_true(3,k+1) = wrapToPi(x_true(3,k+1));    % wrap heading
end

% generate noisy measurements y1 and y2 from the true state at each time step
for k = 1:(K+1)
    y1(:,k) = H*x_true(:,k) + sig_y1 .* randn(ny,1); % sensor 1 output
    y2(:,k) = H*x_true(:,k) + sig_y2 .* randn(ny,1); % sensor 2 output
end
x = sdpvar(nx, N+1, 'full');   % MHE state variables
w = sdpvar(nx, N, 'full');     % process noise variables
x_prior = sdpvar(nx,1);        % prior state
Upar    = sdpvar(nu, N, 'full');% input over horizon
Y1par   = sdpvar(ny, N+1,'full');% sensor 1 data
Y2par   = sdpvar(ny, N+1,'full');% sensor 2 data
R1inv = inv(R1);               % inverse sensor 1 covariance
R2inv = inv(R2);               % inverse sensor 2 covariance
con = [];                      % constraint set
obj = 0;                       % objective function
e0  = x(:,1) - x_prior;        % prior error
obj = obj + e0' * Pinv * e0;   % prior cost

% Build MHE dynamics + process-noise + smoothing terms across the horizon
for kk = 1:N
    con = [con, x(:,kk+1) == A*x(:,kk) + B*Upar(:,kk) + w(:,kk)]; % dynamics
    obj = obj + w(:,kk)' * Qinv * w(:,kk);                      % process cost
    dy = x(2,kk+1) - x(2,kk);                                   % y increment
    obj = obj + lambda_smooth * (dy^2);                         % smoothing
end

% add sensor-fusion measurement residual costs for every node in the horizon
for kk = 1:(N+1)
    r1 = Y1par(:,kk) - H*x(:,kk);  % sensor 1 residual
    r2 = Y2par(:,kk) - H*x(:,kk);  % sensor 2 residual
    obj = obj + r1'*R1inv*r1 + r2'*R2inv*r2; % measurement cost
end

con = [con, -pi <= x(3,:) <= pi]; % heading bounds
con = [con, -x_big <= x(1,:) <= x_big]; % x bounds
con = [con, x(2,:) >= y_wall];    % wall constraint

ops = sdpsettings('solver','quadprog','verbose',0); % solver options
mhe_solver = optimizer(con, obj, ops,{x_prior, Upar, Y1par, Y2par}, {x(:,end), x});  % MHE solver
xhat_mhe = nan(nx, K+1);          % MHE estimate
xhat_mhe(:,1) = [0; 0; 0];        % initial estimate
x_prior_k = xhat_mhe(:,1);        % rolling prior

% Run the MHE online by sliding the window forward and updating the prior each step
for j = N:K
    idx   = (j-N+1):(j+1);        % measurement indices
    idx_u = (j-N+1):j;            % input indices
    [sol, problem] = mhe_solver{x_prior_k,u_true(:,idx_u),y1(:,idx), y2(:,idx)};
    if problem ~= 0
        continue;
    end
    xhat_mhe(:,j+1) = sol{1};     % current estimate
    x_prior_k       = sol{2}(:,2);% shift prior
end
H_aug = [H; H];                   % stacked measurement matrix
R_aug = blkdiag(R1, R2);           % stacked covariance
Q_kf  = zeros(nx);                 % KF process noise
xhat_kf = zeros(nx, K+1);          % KF estimate
xhat_kf(:,1) = [0; 0; 0];          % initial KF state
P = P0;                            % KF covariance
% Run a standard Kalman Filter using the fused (stacked) measurement vector
for k = 1:K
    x_pred = A*xhat_kf(:,k) + B*u_true(:,k); % KF prediction
    P_pred = A*P*A' + Q_kf;                  % covariance prediction
    yk1 = [y1(:,k+1); y2(:,k+1)];             % fused measurement
    S   = H_aug*P_pred*H_aug' + R_aug;        % innovation covariance
    Kg  = (P_pred*H_aug')/S;                  % Kalman gain
    innov = yk1 - H_aug*x_pred;               % innovation
    x_upd = x_pred + Kg*innov;                % state update
    x_upd(3) = wrapToPi(x_upd(3));             % wrap heading
    P = (eye(nx) - Kg*H_aug)*P_pred;           % covariance update
    xhat_kf(:,k+1) = x_upd;                    % save estimate
end
t = (0:K)*dt;                   % time vector
figure; hold on; grid on;
plot(x_true(1,:), x_true(2,:), 'k', 'LineWidth', 1.8);      % true path
plot(xhat_kf(1,:), xhat_kf(2,:), 'b', 'LineWidth', 1.5);   % KF path
plot(xhat_mhe(1,:), xhat_mhe(2,:), 'r', 'LineWidth', 1.5); % MHE path
yline(y_wall,'--','Wall y = -0.01');                        % wall line
xlabel('x [m]'); ylabel('y [m]');
xlim([0,500])
legend('True','KF (fused)','MHE (fused + wall)','Location','best');
title('Trajectory comparison');

function th = wrapToPi(th)
    th = mod(th + pi, 2*pi) - pi; % angle wrap
end
