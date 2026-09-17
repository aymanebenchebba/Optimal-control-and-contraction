%% =======================================================================
%  Incremental (contraction) LMI synthesis for a discrete-time Lur'e system
% -------------------------------------------------------------------------
%  Numerical companion to:
%     <A.Benchebba, "Optimal controls with incremental ISS guarantees for systems with globally Lipschitz nonlinearities", Venue, Year>   
%     DOI / arXiv / URL: <link>                    % TODO: fill in your link
%
%  This script implements the Proposition 4 (verification of the LMI) and synthesizes a 
%  state-feedback law u = -K*x - Gamma*phi(x) that renders the closed-loop Lur'e-type
%  discrete-time system
%
%        x+ = A x + B u + H phi(x),      phi 1-Lipschitz (sector bounded)
%
%  contracting.
%
%  Proposition (LMI_change_of_variables). Let eps in (0,1) and
%  Omega in R^{qxn} be such that Sx = Omega'*Omega. The following are
%  equivalent:
%   (i)  There exist P in S_n, R in S_m, both positive definite, and
%        tau >= 0 such that the ORIGINAL condition (eq. LMI_W_contraction)
%        holds.
%   (ii) There exist X in S_n, Y in S_m, both positive definite, and
%        theta in R_+ such that
%
%        [ (1-eps)*X        0          X*A'         X*Omega' ]
%        [    *          theta*Sphi   theta*H'          0     ]   >= 0
%        [    *              *        X+B*Y*B'          0     ]
%        [    *              *            *         theta*I_q ]
%
%  Script outline:
%    1) System data and sector/Lipschitz bound for the nonlinearity
%    2) LMI synthesis (condition (ii) above), solved with YALMIP
%    3) Recovery of (P, R, tau) and of the feedback gains (K*, Gamma*),
%       with an independent numerical check of condition (i)
%    4) Numerical validation of the contraction property:
%         a) unforced trajectory pairs
%         b) large-scale Monte Carlo sweep
%         c) synchronization under a common external disturbance
%    5) Figure
%
%  Requirements: YALMIP + an SDP solver (SeDuMi, SDPT3, MOSEK, ...) on the
%  MATLAB path. The solver is auto-selected by YALMIP below; set one
%  explicitly with sdpsettings('solver','mosek') (or similar) if needed.
% =======================================================================
clear; close all; clc;
rng(0);   % fix the random seed so the examples below are reproducible

%% ------------------------------------------------------------------
%  1) System data: discrete-time Lur'e model  x+ = A x + B u + H phi(x)
% ---------------------------------------------------------------------
% Continuous-time model
a = 1;
b = 1.0;
c = 1.0;
Ac = [0,  1,  0,  0;
     -b,  0,  b,  0;
      0,  0,  0,  1;
      c,  0, -c,  0];
Bc = [0; 0; 0; 1];
Hc = [0; -a; 0; 0];

% Forward-Euler discretization
Ts = 0.01;                    % sampling time
A  = eye(4) + Ts * Ac;
B  = Ts * Bc;
H  = Ts * Hc;
n  = size(A, 1);
m  = size(B, 2);
p  = size(H, 2);

% Sector / incremental Lipschitz bound for phi:
%   |phi(x1) - phi(x2)| <= Lphi * |Cq*(x1-x2)|
% encoded as the quadratic constraint [Dx;Dphi]' * S * [Dx;Dphi] >= 0,
% with S = [Sx, Sxphi; Sxphi', Sphi] and Sx = Omega'*Omega.
Cq    = [1, 0, 0, 0];
Lphi  = 1.0;                  % sin(.) is globally 1-Lipschitz
Sx    = (Lphi^2) * (Cq' * Cq);
Sxphi = zeros(n, p);
Sphi  = 1.0;
S     = [Sx, Sxphi; Sxphi', -Sphi];
Omega = Lphi * Cq;            % Sx = Omega' * Omega, used directly in the LMI
q     = size(Omega, 1);

epsn = 0.4;                  % target contraction rate (1 - epsn)
phi  = @(x) sin(x(1));        % nonlinearity, applied to Cq*x

%% ------------------------------------------------------------------
%  2) LMI synthesis -- condition (ii) of the Proposition
% ---------------------------------------------------------------------
%  Decision variables: X in S_n (pd), Y in S_m (pd), theta in R_+.
%  NOTE on signs: the "theta*Sphi" block below is transcribed exactly as
%  in the Proposition. With Sphi = -1 as defined above, double-check the
%  sign convention used for S in the paper if the solver reports the
%  problem as infeasible.
X     = sdpvar(n, n, 'symmetric');
Y     = sdpvar(m, m, 'symmetric');
theta = sdpvar(1, 1);
marg  = sdpvar(1, 1);          % feasibility margin, maximized below

M = [ (1-epsn)*X,  zeros(n,p),   X*A',          X*Omega';
      zeros(p,n),  theta*Sphi,   theta*H',      zeros(p,q);
      A*X,         theta*H,      X + B*Y*B',    zeros(n,q);
      Omega*X,     zeros(q,p),   zeros(q,n),    theta*eye(q) ];

Constraints = [ M - marg*eye(size(M,1)) >= 0, ...
                X - 1e-6*eye(n) >= 0, ...
                Y - 1e-6*eye(m) >= 0 ];

options = sdpsettings('verbose', 1);
diagnostics = optimize(Constraints, -marg, options);
if diagnostics.problem ~= 0
    warning('YALMIP/solver reported an issue: %s', diagnostics.info);
end

Xv = double(X); thetav = double(theta); Yv = double(Y); margv = double(marg);
fprintf('[LMI]  feasible with margin t = %.4f > 0\n', margv);
assert(margv > 0, ['LMI infeasible: decrease eps, double-check the sign ' ...
    'of S (see note above), or inspect the solver output.']);

%% ------------------------------------------------------------------
%  3) Recovering (P, R, tau) and the feedback gains (K*, Gamma*)
% ---------------------------------------------------------------------
P   = inv(Xv);
Rv  = inv(Yv);           % R = Y^{-1}: recovered a posteriori (see Proposition)
tau = 1 / thetav;

Sigma = Rv + B' * P * B;
Mp    = P - P*B*inv(Sigma)*B'*P;
F     = [A, H];
L     = inv(Sigma)*B'*P*F;
Kstar     = L(:, 1:n);
Gammastar = L(:, n+1:end);

% Independent numerical check of the ORIGINAL condition, i.e. condition
% (i) of the Proposition, using the recovered (P, R, tau).
top = [(1-epsn)*P, zeros(n,p); zeros(p,n), zeros(p,p)];
lhs = top - tau*S - F'*Mp*F;
eigl = eig(lhs);
fprintf('[LMI]  P matrix computed, original condition re-verified (min eig = %.4f)\n', min(eigl));
assert(all(eigl > 0), 'original condition failed!');

%% ------------------------------------------------------------------
%  Closed-loop system and incremental Lyapunov (contraction) function
% ---------------------------------------------------------------------
Acl  = A - B*Kstar;
Hcl  = H - B*Gammastar;
f_cl = @(x) Acl * x + Hcl * sin(Cq * x);                       % works for a single state (n x 1) or a batch (n x N)
Wfun = @(x1, x2) sum((x1 - x2) .* (P * (x1 - x2)), 1);         % vectorized quadratic form V(x1,x2), returns 1 x N

%% ------------------------------------------------------------------
%  4) Numerical validation of the contraction property
% ---------------------------------------------------------------------
% 4a) Unforced trajectory pairs: check that W(x1_k, x2_k) contracts at
%     the prescribed rate (1-eps) along simulated pairs of trajectories.
T = 500;
n_pairs = 12;
traj1 = zeros(n, T+1, n_pairs);
traj2 = zeros(n, T+1, n_pairs);

for i = 1:n_pairs
    traj1(:, 1, i) = -6 + 12*rand(n, 1);
    traj2(:, 1, i) = -6 + 12*rand(n, 1);
    for k = 1:T
        traj1(:, k+1, i) = f_cl(traj1(:, k, i));
        traj2(:, k+1, i) = f_cl(traj2(:, k, i));
    end
end

Wk = zeros(n_pairs, T+1);
for i = 1:n_pairs
    Wk(i, :) = Wfun(squeeze(traj1(:, :, i)), squeeze(traj2(:, :, i)));
end

ratios = Wk(:, 2:end) ./ Wk(:, 1:end-1);
ratios(Wk(:, 1:end-1) <= 1e-10) = NaN;
worst_ratio_traj = max(ratios(:), [], 'omitnan');

fprintf('\n[check A - along trajectories, converging to the origin]\n');
fprintf('  worst-case ratio observed = %.6f   (must be <= 1-eps = %.6f)\n', worst_ratio_traj, 1-epsn);
if worst_ratio_traj <= 1-epsn+1e-9
    fprintf('  -> PASSED\n');
else
    fprintf('  -> FAILED\n');
end

% 4b) Large-scale Monte Carlo sweep over random pairs in a wide box.
N = 200000;
x1s = -20 + 40*rand(n, N);
x2s = -20 + 40*rand(n, N);
mask = sqrt(sum((x1s - x2s).^2, 1)) > 1e-9;
x1s = x1s(:, mask);
x2s = x2s(:, mask);

W0 = Wfun(x1s, x2s);
W1 = Wfun(f_cl(x1s), f_cl(x2s));
ratio_mc = W1 ./ W0;
worst_ratio_mc = max(ratio_mc);
frac_violating = mean(ratio_mc > 1-epsn+1e-9);

fprintf('\n[check B - Monte Carlo, N=%d random pairs in 4D [-20,20]^4]\n', size(x1s, 2));
fprintf('  worst-case ratio = %.6f  (must be <= 1-eps = %.6f)\n', worst_ratio_mc, 1-epsn);
fprintf('  fraction of sampled pairs violating the bound: %.2e\n', frac_violating);
if worst_ratio_mc <= 1-epsn+1e-9
    fprintf('  -> PASSED\n');
else
    fprintf('  -> FAILED\n');
end

% 4c) Synchronization under a common external disturbance: two
%     trajectories driven by the same forcing signal should still
%     contract towards each other (incremental / synchronization property).
T = 200;
kk_forcing = 0:T-1;
w = 1*sin(0.1*kk_forcing) + 1*cos(0.1*kk_forcing);
x1f = zeros(n, T+1); x2f = zeros(n, T+1);
x1f(:,1) = [1; 0; 1; -1]; %initial condition 1
x2f(:,1) = [-1; 1; -1; 1]; %initial condition 2
for k = 1:T
    x1f(:,k+1) = f_cl(x1f(:,k)) + [0 1 0 0]'*w(k);
    x2f(:,k+1) = f_cl(x2f(:,k)) + [0 1 0 0]'*w(k);
end
Wf = Wfun(x1f, x2f);
ratios_f = Wf(2:end) ./ Wf(1:end-1);
ratios_f(Wf(1:end-1) <= 1e-10) = NaN;
worst_ratio_f = max(ratios_f, [], 'omitnan');

fprintf('\n[check C - synchronization under common disturbance]\n');
fprintf('  worst-case ratio observed = %.6f   (must be <= 1-eps = %.6f)\n', worst_ratio_f, 1-epsn);
if worst_ratio_f <= 1-epsn+1e-9
    fprintf('  -> PASSED\n');
else
    fprintf('  -> FAILED\n');
end

%% ------------------------------------------------------------------
%  5) Figure: state trajectories of the two disturbed solutions
% ---------------------------------------------------------------------
figure('Position', [250 100 800 600]);
tlo = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

t_axis = 0:T;
if size(x1f, 1) ~= 4
    x1f = x1f.';
    x2f = x2f.';
end

colors = lines(4);   % one distinct color per state component

for i = 1:4
    nexttile; hold on; grid on;
    % plot x1 (solid) and x2 (dashed) for component i, same color for both
    plot(t_axis, x1f(i, :), '-',  'Color', colors(i, :), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('$x_{%d}$', i));
    plot(t_axis, x2f(i, :), '--', 'Color', colors(i, :), 'LineWidth', 1.5, ...
        'DisplayName', sprintf("$x'_{%d}$", i));
    xlabel('Time step $k$', 'Interpreter', 'latex');
    legend('Location', 'best', 'Interpreter', 'latex', 'FontSize', 20);
    hold off;
end

% Vector PDF export (tight crop), suitable for direct inclusion in LaTeX
exportgraphics(gcf, 'state_trajectories.pdf', 'ContentType', 'vector', 'BackgroundColor', 'none');
saveas(gcf, 'state_trajectories.png');

%% ------------------------------------------------------------------
%  Continuous-time cross-check (for cross-referencing)
% ---------------------------------------------------------------------
Sigma_c = Rv + Bc' * P * Bc;
Fc = [Ac, Hc];
W_check = [P, zeros(size(P,1), size(Hc,2)); ...
           zeros(size(Hc,2), size(P,1)), zeros(size(Hc,2))] ...
          - Fc' * (P - P*Bc*inv(Sigma_c)*Bc'*P) * Fc;
