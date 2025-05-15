function [config, arms, regret_arrays, n_selected_arrays, timing_stats] = initialize_system()
    % Configuration
    config = struct();
    config.T = 1000;           % Horizon
    config.alpha = 2;          % UCB parameter
    config.epsilon = 0.1;      % Epsilon-greedy parameter
    config.gamma = 0.1;        % EXP3 parameter
    config.V = 3.3;           % Voltage
    config.penalty = 1000;     % Penalty
    config.n_runs = 100;       % Number of runs
    config.N_payload = 10;     % Payload size
    config.CRC = 16;          % CRC size
    config.CR = 1;            % Coding rate
    config.H = 0;             % Header size
    config.d = 50;            % Distance
    
    % Calculate max cost
    SF = 12; BW = 203; Tx = 12.5;
    I = interp1([-18, -12, -3, 3, 9, 12.5], [6.2, 6.8, 8.8, 12.0, 17.7, 24.0], Tx);
    T_sym = (2^SF) / (BW * 1e3);
    N_sym = 8 + 4.25 + 8 + ceil(max(8*config.N_payload + config.CRC - 4*SF + 8 + config.H, 0)/(4*(SF-2))) * (config.CR+4);
    ToA = T_sym * N_sym;
    E = I * config.V * ToA;
    config.cost_max = E * 8 + config.penalty;
    
    % Initialize arms
    SF_vals = [5 7 9 12];
    Tx_vals = [-18 0 12.5];
    BW_vals = [203 812];
    [SF_grid, Tx_grid, BW_grid] = ndgrid(SF_vals, Tx_vals, BW_vals);
    arms = [SF_grid(:), Tx_grid(:), BW_grid(:)];
    
    % Initialize result arrays
    a = size(arms, 1);
    regret_arrays = struct(...
        'ucb', zeros(config.n_runs, config.T), ...
        'eps', zeros(config.n_runs, config.T), ...
        'exp3', zeros(config.n_runs, config.T));
    
    n_selected_arrays = struct(...
        'ucb', zeros(config.n_runs, a), ...
        'eps', zeros(config.n_runs, a), ...
        'exp3', zeros(config.n_runs, a));
    
    % Initialize timing structure
    timing_stats = struct(...
        'ucb_times', zeros(config.n_runs, 1), ...
        'eps_times', zeros(config.n_runs, 1), ...
        'exp3_times', zeros(config.n_runs, 1), ...
        'true_reward_times', zeros(config.n_runs, 1));
end