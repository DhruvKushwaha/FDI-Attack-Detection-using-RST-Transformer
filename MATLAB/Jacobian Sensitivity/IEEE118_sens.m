%% --- MATPOWER setup & solve PF ---
clear all; clc;
define_constants;  % adds BUS_*, GEN_* constants
mpc = loadcase('case118');

% Quiet PF; enforce Q-lims if supported by your MATPOWER version
mpopt = mpoption('verbose', 0, 'out.all', 0);
try
    mpopt = mpoption(mpopt, 'pf.enforce_q_lims', 1);
catch
    % older versions may not support pf.enforce_q_lims; that's fine
end

results = runpf(mpc, mpopt);
if ~results.success
    error('Power flow did not converge.');
end

%% --- Ybus and derivatives dS/dVa, dS/dVm at the solved point ---
baseMVA = results.baseMVA;
bus     = results.bus;
branch  = results.branch;
Vmag    = bus(:, VM);
Vang    = deg2rad(bus(:, VA));
V       = Vmag .* exp(1j * Vang);

[Ybus, ~, ~] = makeYbus(baseMVA, bus, branch);
[dS_dVa, dS_dVm] = dSbus_dV(Ybus, V);  % nb x nb complex matrices

% Jacobian blocks (no minus sign here;
% these map voltage changes to injection changes)
J11_full = real(dS_dVa);  % dP/dVa
J12_full = real(dS_dVm);  % dP/dVm
J21_full = imag(dS_dVa);  % dQ/dVa
J22_full = imag(dS_dVm);  % dQ/dVm

%% --- Index sets ---
nb  = size(bus,1);
sl  = find(bus(:, BUS_TYPE) == REF);
pv  = find(bus(:, BUS_TYPE) == PV);
pq  = find(bus(:, BUS_TYPE) == PQ);
pvpq = [pv; pq];                 % state angles
npq  = length(pq);
npvpq = length(pvpq);

% Reduced Jacobian used for [dVa; dVm_PQ] response
J11 = J11_full(pvpq, pvpq);
J12 = J12_full(pvpq, pq);
J21 = J21_full(pq,   pvpq);
J22 = J22_full(pq,   pq);
J   = [J11 J12; J21 J22];

%% --- Build S: d|V|/dP (p.u./MW), columns for all buses (slack col will be zero) ---
S = zeros(nb, nb);              % initialize
dP_pu = 1 / baseMVA;            % 1 MW injection as per-unit on system base

% Map from bus index -> column in P equations (only for PV & PQ, not slack)
colmap = zeros(nb,1);  % 0 for slack (no equation), >0 for pvpq position
colmap(pvpq) = 1:npvpq;

for j = 1:nb
    rhs = zeros(npvpq + npq, 1);
    cj = colmap(j);          % column index for P@bus j in reduced equations
    if cj > 0
        rhs(cj) = dP_pu;     % ΔP at bus j (per-unit); ΔQ_PQ = 0
        dx = J \ rhs;        % [dVa; dVm_PQ]
        dVm_pq = dx(npvpq+1:end);  % only PQ magnitudes move
        dVmag  = zeros(nb,1);
        dVmag(pq) = dVm_pq;       % PV & slack magnitudes fixed (≈0)
        S(:, j) = dVmag;          % p.u. of |V| per 1 p.u. P
    else
        % j is slack: not part of P equations; leave column as zeros
        S(:, j) = 0;
    end
end

% Convert from per 1 p.u. P to per 1 MW:
S = S;  % we injected dP_pu already, so S is p.u./MW as desired

%% --- Rankings (descending) ---
% 1) Vulnerability of affected buses (row L1 across non-slack columns)
rowL1 = sum(abs(S(:, pvpq)), 2);
[vals_vuln, idx_vuln] = sort(rowL1, 'descend');
types = repmat("PQ", nb, 1);
types(pv) = "PV";
types(sl) = "Slack";
vuln_rank = table(idx_vuln, types(idx_vuln), vals_vuln, ...
    'VariableNames', {'Bus', 'Type', 'Vulnerability_L1_pu_per_MW'});

% 2) Injection influence (how much each injection bus moves voltages overall)
colL1 = sum(abs(S), 1).';
[vals_inj, idx_inj] = sort(colL1, 'descend');
inj_types = types;
inj_influence_rank = table(idx_inj, inj_types(idx_inj), vals_inj, ...
    'VariableNames', {'InjectionBus', 'Type', 'Influence_L1_pu_per_MW'});

% 3) Self-sensitivity diag(S) (PQ buses only; PV/Slack are ~0 by construction)
dself = diag(S);
[dself_vals, dself_idx] = sort(abs(dself), 'descend');
self_sens_rank = table(dself_idx, types(dself_idx), dself(dself_idx), abs(dself_vals), ...
    'VariableNames', {'Bus', 'Type', 'SelfSens_pu_per_MW', 'AbsSelfSens'});

%% --- Pretty print (descending) ---
fprintf('\n=== Vulnerable buses by |dV| sensitivity (row L1), descending ===\n');
fprintf('   Bus   Type    Sum_j |d|V_i|/dP_j|   (p.u./MW)\n');
for k = 1:nb
    fprintf('%6d  %6s   %12.6g\n', vuln_rank.Bus(k), vuln_rank.Type(k), vuln_rank.Vulnerability_L1_pu_per_MW(k));
end

fprintf('\n=== Injection buses by overall influence (col L1), descending ===\n');
fprintf('   Bus   Type    Sum_i |d|V_i|/dP_j|   (p.u./MW)\n');
for k = 1:nb
    fprintf('%6d  %6s   %12.6g\n', inj_influence_rank.InjectionBus(k), inj_influence_rank.Type(k), inj_influence_rank.Influence_L1_pu_per_MW(k));
end

fprintf('\n=== Self-sensitivity d|V_bus|/dP_bus (signed), descending by |.| ===\n');
fprintf('   Bus   Type    d|V|/dP   (p.u./MW)\n');
for k = 1:nb
    fprintf('%6d  %6s   % .6e\n', self_sens_rank.Bus(k), self_sens_rank.Type(k), self_sens_rank.SelfSens_pu_per_MW(k));
end
