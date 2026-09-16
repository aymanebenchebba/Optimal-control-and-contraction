

clear; close all; clc;
rng(0);
% 
% %% 1) Scalar Lur'e example:  x+ = A*x + B*u + H*phi(x),  phi 1-Lipschitz
% A = 0.9;
% B = 1.0;
% H = 0.35;
% n = size(A,1); m = size(B,2); p = size(H,2);
% 
% R = 1.0;
% Lphi = 1.0;
% Sx = Lphi^2; Sxphi = 0.0; Sphi = -1.0;      % sector matrix S
% S = [Sx, Sxphi; Sxphi', Sphi];   % S_xphi' matters once S_xphi is a matrix (n>1 or p>1)
% Omega = Lphi;                                    % Sx = Omega' * Omega
% q = size(Omega,1);
% epsn = 0.15;                                      % contraction rate target (avoid name "eps": MATLAB builtin)
% 
% phi = @(x) tanh(x);    % globally 1-Lipschitz: |phi(x1)-phi(x2)| <= |x1-x2|
% 
% %% 2) Solve eq. "LMI_change_of_variables" with YALMIP
% X     = sdpvar(n, n, 'symmetric');
% theta = sdpvar(1, 1);
% Y     = sdpvar(m, n, 'full');
% Z     = sdpvar(m, p, 'full');
% t     = sdpvar(1, 1);                 % feasibility margin, maximized below
% 
% Rinv  = inv(R);
% Acal  = A*X - B*Y;
% Hcal  = theta*H - B*Z;
%% Numerical Example
% Parameters
a = 1;
b = 1.0;
c = 1.0;
% Continuous system
Ac = [0,  1,  0,  0;
     -b,  0,  b,  0;
      0,  0,  0,  1;
      c,  0, -c,  0];
Bc = [0; 0; 0; 1];
Hc = [0; -a; 0; 0]; 
% Discretization
Ts = 0.01; % time step
A = eye(4) + Ts * Ac;
B = Ts * Bc;
H = Ts * Hc;
n = size(A,1); m = size(B,2); p = size(H,2);
R = 1.0;
Lphi = 1.0; % sin is 1 Lipschitz 
Cq = [1, 0, 0, 0]; 
Sx = (Lphi^2) * (Cq' * Cq); 
Sxphi = zeros(n, p);         
Sphi = -1.0;              
S = [Sx, Sxphi; Sxphi', Sphi]; 
Omega = Lphi * Cq;       % for the LMI
q = size(Omega,1);
epsn = 0.01;             % Contraction rate target
phi = @(x) sin(x(1));     

%% 2) Solve LMI
X     = sdpvar(n, n, 'symmetric');
theta = sdpvar(1, 1);
Y     = sdpvar(m, n, 'full');
Z     = sdpvar(m, p, 'full');
t     = sdpvar(1, 1);                 % feasibility margin, maximized below
Rinv  = inv(R);
Acal  = A*X - B*Y;
Hcal  = theta*H - B*Z;
M = [ (1-epsn)*X,   -X*Sxphi,        Acal',       Y',          X*Omega';
      -Sxphi'*X,    -theta*Sphi,  Hcal',       Z',          zeros(p,q);
      Acal,          Hcal,           X,           zeros(n,m),  zeros(n,q);
      Y,             Z,              zeros(m,n),  Rinv,        zeros(m,q);
      Omega*X,       zeros(q,p),     zeros(q,n),  zeros(q,m),  theta*eye(q) ];
Constraints = [ M - t*eye(size(M,1)) >= 0, ...
                X - 1e-6*eye(n) >= 0 ];
options = sdpsettings('verbose', 1);
diagnostics = optimize(Constraints, -t, options);
if diagnostics.problem ~= 0
    warning('YALMIP/solver reported an issue: %s', diagnostics.info);
end
Xv = double(X); thetav = double(theta); Yv = double(Y); Zv = double(Z); tv = double(t);
fprintf('[LMI]  feasible with margin t = %.4f > 0\n', tv);
assert(tv > 0, 'LMI infeasible: decrease eps, check S_xx>=0, or inspect the solver output above.');

%% 3) Recovering (P, tau) K*, Gamma* = Sigma^-1 B' P F
P   = inv(Xv);
tau = 1/thetav;
Sigma = R + B'*P*B;
Mp    = P - P*B*inv(Sigma)*B'*P;
F     = [A, H];
L     = inv(Sigma)*B'*P*F;
Kstar     = L(:, 1:n);
Gammastar = L(:, n+1:end);

% independent re-check of the ORIGINAL condition eq. "schur_form"
top = [(1-epsn)*P, zeros(n,p); zeros(p,n), zeros(p,p)];
lhs = top - tau*S - F'*Mp*F;
eigl = eig(lhs);
fprintf('[LMI]  P matrix computed, original condition re-verified (min eig = %.4f)\n', min(eigl));
assert(all(eigl > 0), 'original condition failed!');

Acl = A - B*Kstar;
Hcl = H - B*Gammastar;
f_cl = @(x) Acl * x + Hcl * sin(Cq * x); % Supporte vecteurs (nx1) et matrices (nxN)
Wfun = @(x1, x2) sum((x1 - x2) .* (P * (x1 - x2)), 1); % Quadratique vectorisé (1xN)

%% 4a) Simulated pairs 
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

%% 4b) Montecarlo
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

%% 4c) Disturbance
T = 400;
kk_forcing = 0:T-1;
w = 0.1*sin(0.1*kk_forcing) + 0.1*cos(0.1*kk_forcing);   
x1f = zeros(n, T+1); x2f = zeros(n, T+1);
x1f(:,1) = [1; 0; 1; -1]; 
x2f(:,1) = [-1; 1; 0; 1];
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

% %% 5) Plots
% figure('Position', [100 100 1100 800]);
% 
% % Subplot 1: Premier état (x_1) pour illustrer les trajectoires 4D des paires
% subplot(2,2,1); hold on;
% for i = 1:n_pairs
%     plot(0:T, squeeze(traj1(1, :, i)), '-');
%     plot(0:T, squeeze(traj2(1, :, i)), '--');
% end
% xlabel('Time step $k$', 'Interpreter', 'latex'); ylabel('state $x_1$', 'Interpreter', 'latex','FontSize',12);
% title('12 Unforced pairs ($x_1$ component)', 'Interpreter', 'latex', 'FontSize', 14);
% hold off;
% 
% % Subplot 2: Décroissance V
% subplot(2,2,2); hold on;
% kk = 0:T;
% light_blue = [0.6 0.75 0.92];
% for i = 1:n_pairs
%     hsim1 = semilogy(kk, Wk(i,:)/Wk(i,1) + 1e-16, '-', 'Color', light_blue);
% end
% hbound1 = semilogy(kk, (1-epsn).^kk, 'k--', 'LineWidth', 2);
% xlabel('Time step $k$', 'Interpreter', 'latex'); ylabel('$V(x_1^k,x_2^k)/V(x_1^0,x_2^0)$', 'Interpreter', 'latex', 'FontSize', 12);
% title('Incremental distance vs. theoretical rate', 'Interpreter', 'latex', 'FontSize', 14);
% legend([hsim1, hbound1], {'12 Simulations', '$(1-\epsilon)^k$ bound'}, 'Location', 'best', 'Interpreter', 'latex');
% hold off;
% 
% subplot(2,2,3); hold on;
% delta_xf = x1f - x2f; % Calcul de l'incrément entre les deux trajectoires
% plot(0:T, delta_xf, '-', 'LineWidth', 1.5);
% grid on;
% xlabel('Time step $k$', 'Interpreter', 'latex'); 
% ylabel('$\Delta x(k) = x_{1}(k) - x_{2}(k)$', 'Interpreter', 'latex', 'FontSize', 12);
% title('Error dynamics under common forcing', 'Interpreter', 'latex', 'FontSize', 14);
% legend({'$\Delta x_1$', '$\Delta x_2$', '$\Delta x_3$', '$\Delta x_4$'}, 'Location', 'best', 'Interpreter', 'latex');
% hold off;
% 
% % Subplot 4: Décroissance V sous forçage
% subplot(2,2,4); hold on;
% hsim2 = semilogy(0:T, Wf/Wf(1) + 1e-16, 'r-', 'LineWidth', 1.5);
% hbound2 = semilogy(0:T, (1-epsn).^(0:T), 'k--', 'LineWidth', 2);
% xlabel('Time step $k$', 'Interpreter', 'latex'); ylabel('$V(x_1^k,x_2^k)/V(x_1^0,x_2^0)$', 'Interpreter', 'latex', 'FontSize', 12);
% title('Incremental distance under common forcing', 'Interpreter', 'latex', 'FontSize', 14);
% legend([hsim2, hbound2], {'Simulation', '$(1-\epsilon)^k$ bound'}, 'Location', 'best', 'Interpreter', 'latex');
% hold off;
% sgtitle('Contraction verification of the closed loop');
% saveas(gcf, 'contraction_check.png');
% fprintf('\nPlot saved to contraction_check.png\n');

%% 6) Logarithmic scale 
figure('Position', [250 100 800 600]);

% Utilisation de tiledlayout pour minimiser les marges et l'espacement
tlo = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

t = 0:T;
% S'assure que les matrices sont bien de taille (4 x T+1)
if size(x1f, 1) ~= 4
    x1f = x1f.';
    x2f = x2f.';
end

% Palette de 4 couleurs distinctes (une par composante)
colors = lines(4); 

for i = 1:4
    nexttile; % Remplace la commande subplot(2, 2, i)
    hold on; grid on;
    
    % Trace x1 (ligne continue) et x2 (ligne pointillée) avec LA MÊME couleur
    plot(t, x1f(i, :), '-',  'Color', colors(i, :), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('$x_{%d}$', i));
    plot(t, x2f(i, :), '--', 'Color', colors(i, :), 'LineWidth', 1.5, ...
        'DisplayName', sprintf("$x'_{%d}$", i));
    
    xlabel('Time step $k$', 'Interpreter', 'latex');
    legend('Location', 'best', 'Interpreter', 'latex', 'FontSize', 20); % Taille de police légèrement réduite pour un rendu optimal 
    hold off;
end

% Sauvegarde de la figure avec un recadrage automatique parfait pour LaTeX
% Le format PDF vectoriel est recommandé pour conserver la qualité des polices LaTeX
exportgraphics(gcf, 'ma_figure.pdf', 'ContentType', 'vector', 'BackgroundColor', 'none');
% Titre global
% sgtitle('State dynamics under common disturbance', 'Interpreter', 'latex', 'FontSize', 14);

% Sauvegarde de l'image
saveas(gcf,'evolution_between_error_trajectory.png');

% figure('Position', [250 250 700 500]);
% hold on; grid on;
% % for i = 1:n_pairs
% %     hsim_unforced = semilogy(0:T, Wk(i,:) / Wk(i,1), '-', 'Color', light_blue, 'LineWidth', 1);
% % end
% hsim_f = semilogy(0:T, Wf / Wf(1), 'r-', 'LineWidth', 1.5);
% hbound = semilogy(0:T, (1-epsn).^(0:T), 'k--', 'LineWidth', 2);
% xlabel('Time step $k$', 'Interpreter', 'latex');
% ylabel('$\frac{V(x_1^k, x_2^k)}{V(x_1^0, x_2^0)}$', 'Interpreter', 'latex', 'FontSize', 12);
% title('Incremental decrease of $V$', 'Interpreter', 'latex', 'FontSize', 14);
% legend([hsim_f, hbound], ...
%        {'Common forcing', sprintf('Theoretical rate $(1-\\epsilon)^k$')}, ...
%        'Location', 'southwest', 'Interpreter', 'latex');
% xlim([0 T]);
% set(gca, 'YScale', 'log'); 
% hold off;
% saveas(gcf, 'contraction_decay_V.png');


%% Information for latex 
% Define Sigma
Sigma = R + Bc' * P * Bc;

% Define F = [A H]
F = [Ac Hc];

% Compute W
W = [P, zeros(size(P,1), size(Hc,2));
     zeros(size(Hc,2), size(P,1)), zeros(size(Hc,2))] ...
    - F' * (P - P*Bc*(Sigma\ (Bc'*P))) * F;