% ----------------------------
% LoRa Parameter Optimization using Multi-Armed Bandits
% Author: Wany917
% Last Updated: 2025-05-15 12:47:32 UTC
% ----------------------------
clear all;
close all;
clc;

fprintf('Starting execution at 2025-05-15 12:47:32 UTC\n');
fprintf('User: Wany917\n\n');

% ----------------------------
% Configuration
% ----------------------------
config = struct();
config.T = 1000;           % Horizon
config.alpha = 2;          % Exploration boost for UCB
config.epsilon = 0.1;      % For epsilon-greedy
config.gamma = 0.1;        % Learning rate for EXP3
config.V = 3.3;           % Voltage
config.penalty = 1000;     % Penalty for failed transmission
config.n_runs = 100;       % Number of runs
config.N_payload = 10;     % Payload size
config.CRC = 16;          % CRC size
config.CR = 1;            % Coding rate
config.H = 0;             % Header size
config.d = 50;            % Distance
config.cost_max = compute_cost_max(config.penalty);
config.timestamp = datetime('now', 'TimeZone', 'UTC', 'Format', 'yyyy-MM-dd HH:mm:ss');

% ----------------------------
% Initialize arms
% ----------------------------
SF_vals = [5 7 9 12];
Tx_vals = [-18 0 12.5];
BW_vals = [203 812];
[SF_grid, Tx_grid, BW_grid] = ndgrid(SF_vals, Tx_vals, BW_vals);
arms = [SF_grid(:), Tx_grid(:), BW_grid(:)];
a = size(arms, 1);

% ----------------------------
% Initialize result structures
% ----------------------------
regret_ucb_all = zeros(config.n_runs, config.T);
regret_eps_all = zeros(config.n_runs, config.T);
regret_exp3_all = zeros(config.n_runs, config.T);
n_selected_ucb_all = zeros(config.n_runs, a);
n_selected_eps_all = zeros(config.n_runs, a);
n_selected_exp3_all = zeros(config.n_runs, a);

% Initialize timing structures
timing_stats = struct();
timing_stats.ucb_times = zeros(config.n_runs, 1);
timing_stats.eps_times = zeros(config.n_runs, 1);
timing_stats.exp3_times = zeros(config.n_runs, 1);
timing_stats.true_reward_times = zeros(config.n_runs, 1);

% ----------------------------
% Run experiments
% ----------------------------
fprintf('Running %d experiments with %d rounds each...\n', config.n_runs, config.T);
for run = 1:config.n_runs
    fprintf('Running iteration %d/%d...\n', run, config.n_runs);
    rng(run);
    
    % Calculate true rewards with timing
    tic;
    true_rewards = calculate_true_rewards(arms, config);
    timing_stats.true_reward_times(run) = toc;
    max_true_reward = max(true_rewards);
    
    % Run each algorithm with timing
    tic;
    [regret_ucb, n_selected_ucb] = run_ucb(arms, config, max_true_reward);
    timing_stats.ucb_times(run) = toc;
    
    tic;
    [regret_eps, n_selected_eps] = run_epsilon_greedy(arms, config, max_true_reward);
    timing_stats.eps_times(run) = toc;
    
    tic;
    [regret_exp3, n_selected_exp3] = run_exp3(arms, config, max_true_reward);
    timing_stats.exp3_times(run) = toc;
    
    % Store results
    regret_ucb_all(run, :) = regret_ucb;
    regret_eps_all(run, :) = regret_eps;
    regret_exp3_all(run, :) = regret_exp3;
    n_selected_ucb_all(run, :) = n_selected_ucb;
    n_selected_eps_all(run, :) = n_selected_eps;
    n_selected_exp3_all(run, :) = n_selected_exp3;
end

% Calculate timing statistics
timing_stats.mean_times = struct(...
    'ucb', mean(timing_stats.ucb_times), ...
    'eps', mean(timing_stats.eps_times), ...
    'exp3', mean(timing_stats.exp3_times), ...
    'true_reward', mean(timing_stats.true_reward_times));
timing_stats.std_times = struct(...
    'ucb', std(timing_stats.ucb_times), ...
    'eps', std(timing_stats.eps_times), ...
    'exp3', std(timing_stats.exp3_times), ...
    'true_reward', std(timing_stats.true_reward_times));

% Calculate statistics
stats = calculate_statistics(regret_ucb_all, regret_eps_all, regret_exp3_all, ...
    n_selected_ucb_all, n_selected_eps_all, n_selected_exp3_all);
stats.timing = timing_stats;

% Generate plots
generate_plots(stats, arms, config);

% Save results
results = struct('config', config, 'stats', stats, 'timestamp', datestr(now, 'yyyy-mm-dd_HH-MM-SS'));
save(sprintf('bandit_results_%s.mat', results.timestamp), 'results');

fprintf('\nExecution completed at %s UTC\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));

% ----------------------------
% Helper Functions
% ----------------------------
function [regret, n_selected] = run_ucb(arms, config, max_true_reward)
    a = size(arms, 1);
    n_selected = zeros(1, a);
    empirical_mean = zeros(1, a);
    ucb_scores = zeros(1, a);
    regret = zeros(1, config.T);
    
    % Initial exploration
    for t = 1:a
        [reward, ~] = simulate_transmission(arms(t,:), config);
        n_selected(t) = 1;
        empirical_mean(t) = reward;
        ucb_scores(t) = reward + sqrt((config.alpha * log(a)) / (2 * n_selected(t)));
        regret(t) = max_true_reward - reward;
    end
    
    % Main loop
    for t = (a+1):config.T
        [~, arm_idx] = max(ucb_scores);
        [reward, ~] = simulate_transmission(arms(arm_idx,:), config);
        n_selected(arm_idx) = n_selected(arm_idx) + 1;
        empirical_mean(arm_idx) = update_empirical_mean(empirical_mean(arm_idx), reward, n_selected(arm_idx));
        ucb_scores(arm_idx) = empirical_mean(arm_idx) + sqrt((config.alpha * log(t)) / (2 * n_selected(arm_idx)));
        regret(t) = max_true_reward - reward;
    end
end

function [regret, n_selected] = run_epsilon_greedy(arms, config, max_true_reward)
    a = size(arms, 1);
    n_selected = zeros(1, a);
    empirical_mean = zeros(1, a);
    regret = zeros(1, config.T);
    
    % Initial exploration
    for t = 1:a
        [reward, ~] = simulate_transmission(arms(t,:), config);
        n_selected(t) = 1;
        empirical_mean(t) = reward;
        regret(t) = max_true_reward - reward;
    end
    
    % Main loop
    for t = (a+1):config.T
        if rand() < config.epsilon
            arm_idx = randi(a);
        else
            [~, arm_idx] = max(empirical_mean);
        end
        
        [reward, ~] = simulate_transmission(arms(arm_idx,:), config);
        n_selected(arm_idx) = n_selected(arm_idx) + 1;
        empirical_mean(arm_idx) = update_empirical_mean(empirical_mean(arm_idx), reward, n_selected(arm_idx));
        regret(t) = max_true_reward - reward;
    end
end

function [regret, n_selected] = run_exp3(arms, config, max_true_reward)
    a = size(arms, 1);
    weights = ones(1, a);
    n_selected = zeros(1, a);
    regret = zeros(1, config.T);
    
    for t = 1:config.T
        prob = (1 - config.gamma) * (weights / sum(weights)) + config.gamma / a;
        arm_idx = randsample(a, 1, true, prob);
        [reward, ~] = simulate_transmission(arms(arm_idx,:), config);
        n_selected(arm_idx) = n_selected(arm_idx) + 1;
        estimated_reward = reward / prob(arm_idx);
        weights(arm_idx) = weights(arm_idx) * exp(config.gamma * estimated_reward / a);
        regret(t) = max_true_reward - reward;
    end
end

function true_rewards = calculate_true_rewards(arms, config)
    n_simulations = 50;
    a = size(arms, 1);
    true_rewards = zeros(1, a);
    
    for i = 1:a
        rewards_sum = 0;
        for j = 1:n_simulations
            [reward, ~] = simulate_transmission(arms(i,:), config);
            rewards_sum = rewards_sum + reward;
        end
        true_rewards(i) = rewards_sum / n_simulations;
    end
end

function [reward_value, transmission_info] = simulate_transmission(arm, config)
    % Define lookup tables
    I_table = [-18, 6.2; -12, 6.8; -3, 8.8; 0, 10.1; 3, 12.0; 9, 17.7; 12.5, 24.0];
    BW_list = [203 406 812 1625];
    SF_list = [5 6 7 8 9 10 11 12];
    sensitivity_table = [
        -109 -111 -115 -118 -121 -124 -127 -130;
        -107 -110 -113 -116 -119 -122 -125 -128;
        -105 -108 -112 -115 -117 -120 -123 -126;
        -99  -103 -106 -109 -111 -114 -117 -120
    ];
    
    SF = arm(1); Tx = arm(2); BW = arm(3);
    I = interp1(I_table(:,1), I_table(:,2), Tx, 'linear', 'extrap');
    T_sym = (2^SF) / (BW * 1e3);
    
    if SF < 7
        N_sym = 8 + 6.25 + 8 + ceil(max(8*config.N_payload + config.CRC - 4*SF + config.H, 0)/(4*SF)) * (config.CR+4);
    elseif SF <= 10
        N_sym = 8 + 4.25 + 8 + ceil(max(8*config.N_payload + config.CRC - 4*SF + 8 + config.H, 0)/(4*SF)) * (config.CR+4);
    else
        N_sym = 8 + 4.25 + 8 + ceil(max(8*config.N_payload + config.CRC - 4*SF + 8 + config.H, 0)/(4*(SF-2))) * (config.CR+4);
    end
    
    ToA = T_sym * N_sym;
    Lp = 40 + 50 * log10(config.d) + 6 + 3;
    G = 2; L = 2;
    P_rx = Tx + G - Lp - L + G - L;
    sf_interp = interp1([5 6 7 8 9 10 11 12], 1:8, SF);
    bw_idx = find(BW_list == BW);
    P_rx_min = interp1(1:8, sensitivity_table(bw_idx,:), sf_interp);
    
    success = false;
    for attempts = 1:8
        noise = -0.5 + rand();
        if P_rx + noise > P_rx_min
            success = true;
            break;
        end
    end
    
    E_transmit = I * config.V * ToA;
    cost = E_transmit * attempts + config.penalty * (~success);
    reward_value = 1 - cost / config.cost_max;
    
    transmission_info = struct('E_transmit', E_transmit, 'cost', cost, ...
        'attempts', attempts, 'ToA', ToA, 'I', I, 'P_rx', P_rx, ...
        'P_rx_min', P_rx_min);
end

function new_mean = update_empirical_mean(prev_mean, new_reward, n)
    new_mean = ((n - 1) * prev_mean + new_reward) / n;
end

function max_cost = compute_cost_max(penalty)
    SF = 12; BW = 203; Tx = 12.5; V = 3.3;
    I = interp1([-18, -12, -3, 3, 9, 12.5], [6.2, 6.8, 8.8, 12.0, 17.7, 24.0], Tx);
    T_sym = (2^SF) / (BW * 1e3);
    N_payload = 10; CRC = 16; CR = 1; H = 0;
    N_sym = 8 + 4.25 + 8 + ceil(max(8*N_payload + CRC - 4*SF + 8 + H, 0)/(4*(SF-2))) * (CR+4);
    ToA = T_sym * N_sym;
    E = I * V * ToA;
    max_cost = E * 8 + penalty;
end

function stats = calculate_statistics(regret_ucb_all, regret_eps_all, regret_exp3_all, ...
    n_selected_ucb_all, n_selected_eps_all, n_selected_exp3_all)
    stats = struct();
    
    % Calculate mean and standard deviation of regrets
    stats.mean_regret_ucb = mean(regret_ucb_all, 1);
    stats.mean_regret_eps = mean(regret_eps_all, 1);
    stats.mean_regret_exp3 = mean(regret_exp3_all, 1);
    
    stats.std_regret_ucb = std(regret_ucb_all, 0, 1);
    stats.std_regret_eps = std(regret_eps_all, 0, 1);
    stats.std_regret_exp3 = std(regret_exp3_all, 0, 1);
    
    % Calculate cumulative statistics
    stats.mean_cum_regret_ucb = cumsum(stats.mean_regret_ucb);
    stats.mean_cum_regret_eps = cumsum(stats.mean_regret_eps);
    stats.mean_cum_regret_exp3 = cumsum(stats.mean_regret_exp3);
    
    stats.std_cum_regret_ucb = cumsum(stats.std_regret_ucb);
    stats.std_cum_regret_eps = cumsum(stats.std_regret_eps);
    stats.std_cum_regret_exp3 = cumsum(stats.std_regret_exp3);
    
    % Calculate mean arm selections
    stats.mean_n_selected_ucb = mean(n_selected_ucb_all, 1);
    stats.mean_n_selected_eps = mean(n_selected_eps_all, 1);
    stats.mean_n_selected_exp3 = mean(n_selected_exp3_all, 1);
end

function generate_plots(stats, arms, config)
    % Cumulative regret
    plot_cumulative_regret(stats, config);
    
    % Average regret per round
    plot_average_regret(stats, config);
    
    % Arm selection frequencies
    plot_arm_selections(stats, arms);
    
    % Algorithm comparison
    plot_algorithm_comparison(stats, config);
    
    % Time performance
    plot_time_performance(stats);
end

function plot_cumulative_regret(stats, config)
    figure('Name', 'Cumulative Regret');
    t = 1:config.T;
    hold on;
    
    % Plot each algorithm with confidence bands
    plot_with_confidence_band(t, stats.mean_cum_regret_ucb, stats.std_cum_regret_ucb, 'b', 'UCB');
    plot_with_confidence_band(t, stats.mean_cum_regret_eps, stats.std_cum_regret_eps, 'r', '\epsilon-greedy');
    plot_with_confidence_band(t, stats.mean_cum_regret_exp3, stats.std_cum_regret_exp3, 'g', 'EXP3');
    
    title('Cumulative Regret over Time');
    xlabel('Round');
    ylabel('Cumulative Regret');
    legend('show');
    grid on;
end

function plot_average_regret(stats, config)
    figure('Name', 'Average Regret per Round');
    t = 1:config.T;
    hold on;
    
    % Calculate average regret per round
    avg_regret_ucb = stats.mean_cum_regret_ucb ./ t;
    avg_regret_eps = stats.mean_cum_regret_eps ./ t;
    avg_regret_exp3 = stats.mean_cum_regret_exp3 ./ t;
    
    std_avg_regret_ucb = stats.std_cum_regret_ucb ./ t;
    std_avg_regret_eps = stats.std_cum_regret_eps ./ t;
    std_avg_regret_exp3 = stats.std_cum_regret_exp3 ./ t;
    
    plot_with_confidence_band(t, avg_regret_ucb, std_avg_regret_ucb, 'b', 'UCB');
    plot_with_confidence_band(t, avg_regret_eps, std_avg_regret_eps, 'r', '\epsilon-greedy');
    plot_with_confidence_band(t, avg_regret_exp3, std_avg_regret_exp3, 'g', 'EXP3');
    
    title('Average Regret per Round');
    xlabel('Round');
    ylabel('Average Regret');
    legend('show');
    grid on;
end

function plot_with_confidence_band(t, mean_data, std_data, color, label)
    plot(t, mean_data, color, 'LineWidth', 1.5, 'DisplayName', label);
    fill([t, fliplr(t)], ...
         [mean_data - std_data, fliplr(mean_data + std_data)], ...
         color, 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
end

function plot_arm_selections(stats, arms)
    labels = arrayfun(@(i) sprintf('SF=%d,Tx=%.1f,BW=%d', ...
        arms(i,1), arms(i,2), arms(i,3)), 1:size(arms,1), ...
        'UniformOutput', false);
    
    figure('Name', 'Arm Selection Frequencies', 'Position', [100 100 1200 800]);
    
    subplot(3,1,1);
    bar(stats.mean_n_selected_ucb);
    title('Average Arm Selections - UCB');
    xlabel('Arm Configuration (SF, Tx, BW)');
    ylabel('Selection Count');
    set(gca, 'XTick', 1:length(labels), 'XTickLabel', labels);
    xtickangle(45);
    grid on;
    
    subplot(3,1,2);
    bar(stats.mean_n_selected_eps);
    title('Average Arm Selections - \epsilon-Greedy');
    xlabel('Arm Configuration (SF, Tx, BW)');
    ylabel('Selection Count');
    set(gca, 'XTick', 1:length(labels), 'XTickLabel', labels);
    xtickangle(45);
    grid on;
    
    subplot(3,1,3);
    bar(stats.mean_n_selected_exp3);
    title('Average Arm Selections - EXP3');
    xlabel('Arm Configuration (SF, Tx, BW)');
    ylabel('Selection Count');
    set(gca, 'XTick', 1:length(labels), 'XTickLabel', labels);
    xtickangle(45);
    grid on;
    
    sgtitle('Arm Selection Frequencies Across Algorithms');
    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
end

function plot_algorithm_comparison(stats, config)
    figure('Name', 'Algorithm Performance Comparison', 'Position', [100 100 1200 800]);
    
    % 1. Final Cumulative Regret Distribution
    subplot(2,2,1);
    final_regrets = [
        stats.mean_cum_regret_ucb(end), ...
        stats.mean_cum_regret_eps(end), ...
        stats.mean_cum_regret_exp3(end)
    ];
    final_stds = [
        stats.std_cum_regret_ucb(end), ...
        stats.std_cum_regret_eps(end), ...
        stats.std_cum_regret_exp3(end)
    ];
    
    b = bar(final_regrets);
    hold on;
    errorbar(1:3, final_regrets, final_stds, 'k.', 'LineWidth', 1.5);
    title('Final Cumulative Regret');
    set(gca, 'XTickLabel', {'UCB', '\epsilon-greedy', 'EXP3'});
    ylabel('Regret Value');
    grid on;
    
    % 2. Convergence Speed
    subplot(2,2,2);
    convergence_threshold = 0.9;
    
    ucb_conv = find_convergence_point(stats.mean_cum_regret_ucb, convergence_threshold);
    eps_conv = find_convergence_point(stats.mean_cum_regret_eps, convergence_threshold);
    exp3_conv = find_convergence_point(stats.mean_cum_regret_exp3, convergence_threshold);
    
    conv_rounds = [ucb_conv, eps_conv, exp3_conv];
    b = bar(conv_rounds);
    title(sprintf('Rounds to %.0f%% Convergence', convergence_threshold*100));
    set(gca, 'XTickLabel', {'UCB', '\epsilon-greedy', 'EXP3'});
    ylabel('Number of Rounds');
    grid on;
    
    % 3. Exploitation vs Exploration
    subplot(2,2,3);
    [~, best_arm] = max(stats.mean_n_selected_ucb);
    exploit_ratio_ucb = stats.mean_n_selected_ucb(best_arm) / sum(stats.mean_n_selected_ucb);
    exploit_ratio_eps = stats.mean_n_selected_eps(best_arm) / sum(stats.mean_n_selected_eps);
    exploit_ratio_exp3 = stats.mean_n_selected_exp3(best_arm) / sum(stats.mean_n_selected_exp3);
    
    exploit_ratios = [exploit_ratio_ucb, exploit_ratio_eps, exploit_ratio_exp3];
    b = bar(exploit_ratios);
    title('Exploitation Ratio');
    set(gca, 'XTickLabel', {'UCB', '\epsilon-greedy', 'EXP3'});
    ylabel('Ratio');
    ylim([0 1]);
    grid on;
    
    % 4. Average Regret in Last 10% of Rounds
    subplot(2,2,4);
    last_idx = round(0.9 * config.T):config.T;
    
    final_avg_regret_ucb = mean(stats.mean_regret_ucb(last_idx));
    final_avg_regret_eps = mean(stats.mean_regret_eps(last_idx));
    final_avg_regret_exp3 = mean(stats.mean_regret_exp3(last_idx));
    
    final_std_regret_ucb = std(stats.mean_regret_ucb(last_idx));
    final_std_regret_eps = std(stats.mean_regret_eps(last_idx));
    final_std_regret_exp3 = std(stats.mean_regret_exp3(last_idx));
    
    final_avg_regrets = [final_avg_regret_ucb, final_avg_regret_eps, final_avg_regret_exp3];
    final_avg_stds = [final_std_regret_ucb, final_std_regret_eps, final_std_regret_exp3];
    
    b = bar(final_avg_regrets);
    hold on;
    errorbar(1:3, final_avg_regrets, final_avg_stds, 'k.', 'LineWidth', 1.5);
    title('Average Regret in Last 10% of Rounds');
    set(gca, 'XTickLabel', {'UCB', '\epsilon-greedy', 'EXP3'});
    ylabel('Average Regret');
    grid on;
    
    sgtitle('Comparative Performance Analysis of Bandit Algorithms');
    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
end

function plot_time_performance(stats)
    figure('Name', 'Time Performance Analysis', 'Position', [100 100 1200 800]);
    
    % 1. Average Execution Time per Algorithm
    subplot(2,2,1);
    times = [stats.timing.mean_times.ucb, ...
             stats.timing.mean_times.eps, ...
             stats.timing.mean_times.exp3];
    stds = [stats.timing.std_times.ucb, ...
            stats.timing.std_times.eps, ...
            stats.timing.std_times.exp3];
    
    b = bar(times);
    hold on;
    errorbar(1:3, times, stds, 'k.', 'LineWidth', 1.5);
    title('Average Execution Time per Algorithm');
    set(gca, 'XTickLabel', {'UCB', '\epsilon-greedy', 'EXP3'});
    ylabel('Time (seconds)');
    grid on;
    
    % 2. Time Distribution
    subplot(2,2,2);
    boxplot([stats.timing.ucb_times, ...
             stats.timing.eps_times, ...
             stats.timing.exp3_times], ...
            'Labels', {'UCB', '\epsilon-greedy', 'EXP3'});
    title('Execution Time Distribution');
    ylabel('Time (seconds)');
    grid on;
    
    % 3. Cumulative Time Performance
    subplot(2,2,3);
    hold on;
    total_times = sort([stats.timing.ucb_times, ...
                       stats.timing.eps_times, ...
                       stats.timing.exp3_times]);
    cdf_y = (1:length(total_times)) / length(total_times);
    
    plot(sort(stats.timing.ucb_times), cdf_y(1:length(stats.timing.ucb_times)), ...
         'b-', 'LineWidth', 1.5, 'DisplayName', 'UCB');
    plot(sort(stats.timing.eps_times), cdf_y(1:length(stats.timing.eps_times)), ...
         'r-', 'LineWidth', 1.5, 'DisplayName', '\epsilon-greedy');
    plot(sort(stats.timing.exp3_times), cdf_y(1:length(stats.timing.exp3_times)), ...
         'g-', 'LineWidth', 1.5, 'DisplayName', 'EXP3');
    
    title('Cumulative Distribution of Execution Times');
    xlabel('Time (seconds)');
    ylabel('Cumulative Probability');
    legend('show');
    grid on;
    
    % 4. Relative Time Performance
    subplot(2,2,4);
    min_time = min([times(1), times(2), times(3)]);
    relative_times = times / min_time;
    relative_stds = stds / min_time;
    
    b = bar(relative_times);
    hold on;
    errorbar(1:3, relative_times, relative_stds, 'k.', 'LineWidth', 1.5);
    title('Relative Time Performance (normalized to fastest)');
    set(gca, 'XTickLabel', {'UCB', '\epsilon-greedy', 'EXP3'});
    ylabel('Relative Time (ratio)');
    grid on;
    
    sgtitle('Time Performance Analysis of Bandit Algorithms');
    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
    
    annotation('textbox', [0.02 0.02 0.3 0.05], ...
        'String', sprintf(['Total Computation Time: %.2fs\n' ...
                          'Fastest Algorithm: %s (%.3fs)\n' ...
                          'Slowest Algorithm: %s (%.3fs)'], ...
                          sum(times), ...
                          get_fastest_algorithm(times), min(times), ...
                          get_slowest_algorithm(times), max(times)), ...
        'EdgeColor', 'none', ...
        'FitBoxToText', 'on');
end

function name = get_fastest_algorithm(times)
    [~, idx] = min(times);
    names = {'UCB', '\epsilon-greedy', 'EXP3'};
    name = names{idx};
end

function name = get_slowest_algorithm(times)
    [~, idx] = max(times);
    names = {'UCB', '\epsilon-greedy', 'EXP3'};
    name = names{idx};
end

function conv_point = find_convergence_point(regret_curve, threshold)
    final_value = regret_curve(end);
    target_value = final_value * threshold;
    conv_point = find(regret_curve >= target_value, 1);
end
