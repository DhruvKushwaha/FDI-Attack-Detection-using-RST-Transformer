%% Load MATPOWER and Set Up the IEEE 9-Bus Case
clear all
clc
define_constants;
mpc = loadcase('case118');  % Load IEEE 9-bus system

%% Set MATPOWER Options to Use Newton-Raphson (NR)
%mpopt = mpoption('PF_ALG', 1);

%% Parameters
num_iterations = 5e4;  
num_buses = size(mpc.bus, 1);
num_branches = size(mpc.branch, 1);
num_gens = size(mpc.gen, 1);
gen_max = mpc.gen(:, 9);
gen_min = mpc.gen(:,10);

% Set percentage attack
per_attack = 0.05;
% Find the generator associated with bus #
gen_idx = mpc.gen(:, 1);

% Error handling if no generator is found
if isempty(gen_idx)
    error('No generator found for the specified bus number.');
end

% Define the noise range (±15% of Pmax)
noise_range = 0.15 * gen_max;

% Generate array `arr` with sine-based noise and Gaussian noise
arr = zeros(num_gens, num_iterations);

% Define MATPOWER options for better stability
mpopt = mpoption('opf.ac.solver', 'MIPS', ...
                 'mips.step_control', 1, ...
                 'opf.violation', 1e-6, ...
                 'mips.feastol', 1e-6, ...
                 'mips.gradtol', 1e-6, ...
                 'mips.comptol', 1e-6);

% Define power demand column index
PD = 3; 

for i = 1:num_iterations
    if rand() > 0.5
        arr(:,i) = (sin(i) + noise_range) + 0.6*gen_max;
    else
        arr(:,i) = (sin(i) - noise_range) + 0.6*gen_max;
    end
end



%% Compute Base-Case Power Flow and Jacobian Sensitivity Analysis
att_buses = 20; % # of buses to attack
%vulnerable_buses = Jacobian_Sens(mpc, mpopt, att_buses);
vulnerable_buses = [21, 43, 20, 44, 52, 22, 51, 53, 45, 58, ...
    108, 117, 109, 86, 33, 57, 95, 13, 19, 87];

%% Run 20 Iterations of FDI + LA Attack Simulation
base_mpc = mpc; 
% Define the total number of samples and attack count
attack_samples = 5e3;
% Prepare Data Storage
data = [];

% Precompute the indices where attacks will happen
attack_indices = randperm(num_iterations, attack_samples); 
% Randomly select 4000 indices for attack

for iter = 1:num_iterations
    fprintf('\nIteration %d...\n', iter);
    mpc=base_mpc;
     % Inject noise value into the bus's real power demand (Pd)
    mpc.bus(:, PD) =  mpc.bus(:, PD) + randn() * 0.05 * mpc.bus(:, PD);

    % Perturb generator setpoints 
    mpc.gen(:, 2) = arr(:, i);
    
    % Initialize attack variables
    attack_bus = 0;  % Default: No attack

    % Check if the current iteration index is in attack_indices
    if ismember(iter, attack_indices)
        % Decide attack bus based on temp value
         attack_bus = datasample(vulnerable_buses, 1, 'Replace', false);

        fprintf('Attacking Bus %d in iteration %d\n', attack_bus, iter);
    else
        fprintf('No attack in iteration %d.\n', iter);
    end

    % Initialize attack_labels correctly
    attack_labels = zeros(length(vulnerable_buses), 1);   %
    % Initialize a zero vector for all buses

    % Get z0
    results_LA = runpf(mpc, mpopt);
    z0 = results_LA.bus(:, PD);

    % If attack occurs, increase load at the selected bus
    if attack_bus > 0
       if rand() > 0.5
           mpc.bus(attack_bus, PD) = mpc.bus(attack_bus, PD) + ...
               per_attack*mpc.bus(attack_bus, PD);
           % Run power flow after the LA attack
           results_LA = runpf(mpc, mpopt);
           z_LA = results_LA.bus(:, PD); 
        
           % Construct the False Data Injection (FDI) Attack Vector
           a_FDI = z0 - z_LA;
           z_FDI = z_LA + a_FDI;  % Manipulated measurements
           results_LA.bus(:, PD) = z_FDI;
       else
           mpc.bus(attack_bus, PD) = mpc.bus(attack_bus, PD) -...
               per_attack*mpc.bus(attack_bus, PD);
           % Run power flow after the LA attack
           results_LA = runpf(mpc, mpopt);
           z_LA = results_LA.bus(:, PD); 
        
           % Construct the False Data Injection (FDI) Attack Vector
           a_FDI = z0 - z_LA;
           z_FDI = z_LA + a_FDI;  % Manipulated measurements
           results_LA.bus(:, PD) = z_FDI;
       end
       % If attack occurs, set attack label to 1 for attacked buses
       if ~isempty(attack_bus)  
           idx = find(vulnerable_buses == attack_bus);
           attack_labels(idx) = ones(length(attack_bus)); 
       end
    end

    % Extract Power Flow Data with Noise

    % Store Data Row
    data_row = [results_LA.branch(:, PF)', results_LA.branch(:, PT)'...
        , results_LA.gen(:, PG)', attack_labels'];
    data = [data; data_row];  % Append to dataset
end

% Generate unique branch numbers for each branch row
branch_numbers = (1:num_branches)';

% Create unique column names based on actual bus numbers and branch indices:
P_from_names = strcat("From_Bus_", string(mpc.branch(:, F_BUS)),...
    "_", string(branch_numbers));
P_to_names   = strcat("To_Bus_", string(mpc.branch(:, T_BUS)),...
    "_", string(branch_numbers));

% Similarly for generators, append an index to ensure uniqueness:
gen_numbers = (1:num_gens)';
Gen_names = strcat("Gen_Bus_", string(mpc.gen(:, GEN_BUS)),...
    "_", string(gen_numbers));

% For buses, we assume the bus numbers are unique.
Attack_names = strcat("Attack_Bus_", string(vulnerable_buses));

% Combine all column names into one row vector:
column_names = [P_from_names.' , P_to_names.' , Gen_names.' , Attack_names];

% Now you can create your table using these unique column names.
data_table = array2table(data, 'VariableNames', column_names);
csv_filename = strcat('./Data/IEEE_case118_onebus_',string(per_attack*100)...
    ,'%.csv');
writetable(data_table, csv_filename);

fprintf('\n✅ Data for one bus saved successfully as %s!\n', csv_filename);
