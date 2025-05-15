% ----------------------------
% LoRa Parameter Optimization using Multi-Armed Bandits
% Author: Wany917
% Last Updated: 2025-05-15 10:44:39 UTC
% ----------------------------
clear all;
close all;
clc;

% Print execution information
fprintf('Starting execution at %s UTC\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf('User: Wany917\n\n');

% Initialize system
[config, arms, regret_arrays, n_selected_arrays, timing_stats] = initialize_system();
a = size(arms, 1);

% Calculate true rewards for each arm
fprintf('Calculating true rewards...\n');
true_rewards = zeros(1, a);
tic;
for i = 1:a
    rewards_temp = zeros(1, 50); % 50 simulations per arm for stable estimate
    for j = 1:50
        [reward, ~] = simulate_transmission(arms(i,:), config);
        rewards_temp(j) = reward;
    end
    true_rewards(i) = mean(rewards_temp);
end
timing_stats.true_reward_time = toc;
max_true_reward = max(true_rewards);

% Run experiments
fprintf('Running %d experiments with %d rounds each...\n', config.n_runs, config.T);
for run = 1:config.n_runs
    fprintf('Run %d/%d\n', run, config.n_runs);
    
    % Set random seed for reproducibility
    rng(run);
    
    % Run UCB
    tic;
    [regret_ucb, n_selected_ucb] = run_bandit_algorithm('ucb', arms, config, max_true_reward);
    timing_stats.ucb_times(run) = toc;
    regret_arrays.ucb(run, :) = regret_ucb;
    n_selected_arrays.ucb(run, :) = n_selected_ucb;
    
    % Run Epsilon-Greedy
    tic;
    [regret_eps, n_selected_eps] = run_bandit_algorithm('epsilon', arms, config, max_true_reward);
    timing_stats.eps_times(run) = toc;
    regret_arrays.eps(run, :) = regret_eps;
    n_selected_arrays.eps(run, :) = n_selected_eps;
    
    % Run EXP3
    tic;
    [regret_exp3, n_selected_exp3] = run_bandit_algorithm('exp3', arms, config, max_true_reward);
    timing_stats.exp3_times(run) = toc;
    regret_arrays.exp3(run, :) = regret_exp3;
    n_selected_arrays.exp3(run, :) = n_selected_exp3;
end

% Calculate statistics
fprintf('\nCalculating statistics...\n');
stats = struct();

% Calculate mean and std of regrets
stats.mean_regret_ucb = mean(regret_arrays.ucb, 1);
stats.mean_regret_eps = mean(regret_arrays.eps, 1);
stats.mean_regret_exp3 = mean(regret_arrays.exp3, 1);

stats.std_regret_ucb = std(regret_arrays.ucb, 0, 1);
stats.std_regret_eps = std(regret_arrays.eps, 0, 1);
stats.std_regret_exp3 = std(regret_arrays.exp3, 0, 1);

% Calculate cumulative statistics
stats.mean_cum_regret_ucb = cumsum(stats.mean_regret_ucb);
stats.mean_cum_regret_eps = cumsum(stats.mean_regret_eps);
stats.mean_cum_regret_exp3 = cumsum(stats.mean_regret_exp3);

stats.std_cum_regret_ucb = cumsum(stats.std_regret_ucb);
stats.std_cum_regret_eps = cumsum(stats.std_regret_eps);
stats.std_cum_regret_exp3 = cumsum(stats.std_regret_exp3);

% Calculate mean arm selections
stats.mean_n_selected_ucb = mean(n_selected_arrays.ucb, 1);
stats.mean_n_selected_eps = mean(n_selected_arrays.eps, 1);
stats.mean_n_selected_exp3 = mean(n_selected_arrays.exp3, 1);

% Store timing information correctly
stats.timing = timing_stats;  % Store the complete timing_stats structure
stats.timing.mean_times = struct(...
    'ucb', mean(timing_stats.ucb_times), ...
    'eps', mean(timing_stats.eps_times), ...
    'exp3', mean(timing_stats.exp3_times));
stats.timing.std_times = struct(...
    'ucb', std(timing_stats.ucb_times), ...
    'eps', std(timing_stats.eps_times), ...
    'exp3', std(timing_stats.exp3_times));

% Generate plots
fprintf('Generating plots...\n');
generate_plots(stats, arms, config);

% Save results
fprintf('Saving results...\n');
timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
results = struct(...
    'config', config, ...
    'stats', stats, ...
    'timestamp', timestamp, ...
    'true_rewards', true_rewards);
save(sprintf('bandit_results_%s.mat', timestamp), 'results');

% Print completion message
fprintf('\nExecution completed at %s UTC\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));

% Print summary statistics
fprintf('\nSummary Statistics:\n');
fprintf('UCB - Average execution time: %.3f s (±%.3f)\n', ...
    stats.timing.mean_times.ucb, stats.timing.std_times.ucb);
fprintf('Epsilon-Greedy - Average execution time: %.3f s (±%.3f)\n', ...
    stats.timing.mean_times.eps, stats.timing.std_times.eps);
fprintf('EXP3 - Average execution time: %.3f s (±%.3f)\n', ...
    stats.timing.mean_times.exp3, stats.timing.std_times.exp3);
fprintf('Final cumulative regret: UCB=%.2f, Eps-Greedy=%.2f, EXP3=%.2f\n', ...
    stats.mean_cum_regret_ucb(end), ...
    stats.mean_cum_regret_eps(end), ...
    stats.mean_cum_regret_exp3(end));