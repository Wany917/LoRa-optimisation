function generate_plots(stats, arms, config)
    % 1. Cumulative Regret
    figure('Name', 'Cumulative Regret');
    t = 1:config.T;
    hold on;
    
    % Plot with confidence bands
    plot_with_bands(t, stats.mean_cum_regret_ucb, stats.std_cum_regret_ucb, 'b', 'UCB');
    plot_with_bands(t, stats.mean_cum_regret_eps, stats.std_cum_regret_eps, 'r', '\epsilon-greedy');
    plot_with_bands(t, stats.mean_cum_regret_exp3, stats.std_cum_regret_exp3, 'g', 'EXP3');
    
    title('Cumulative Regret over Time');
    xlabel('Round'); ylabel('Cumulative Regret');
    legend('show'); grid on;
    
    % 2. Average Regret
    figure('Name', 'Average Regret');
    hold on;
    
    plot_with_bands(t, stats.mean_cum_regret_ucb./t, stats.std_cum_regret_ucb./t, 'b', 'UCB');
    plot_with_bands(t, stats.mean_cum_regret_eps./t, stats.std_cum_regret_eps./t, 'r', '\epsilon-greedy');
    plot_with_bands(t, stats.mean_cum_regret_exp3./t, stats.std_cum_regret_exp3./t, 'g', 'EXP3');
    
    title('Average Regret per Round');
    xlabel('Round'); ylabel('Average Regret');
    legend('show'); grid on;
    
    % 3. Arm Selections
    figure('Name', 'Arm Selection Frequencies', 'Position', [100 100 1200 800]);
    labels = arrayfun(@(i) sprintf('SF=%d,Tx=%.1f,BW=%d', arms(i,1), arms(i,2), arms(i,3)), ...
        1:size(arms,1), 'UniformOutput', false);
    
    % UCB selections
    subplot(3,1,1);
    bar(stats.mean_n_selected_ucb);
    title('Average Arm Selections - UCB');
    set(gca, 'XTick', 1:length(labels), 'XTickLabel', labels);
    xtickangle(45); grid on;
    
    % Epsilon-greedy selections
    subplot(3,1,2);
    bar(stats.mean_n_selected_eps);
    title('Average Arm Selections - \epsilon-greedy');
    set(gca, 'XTick', 1:length(labels), 'XTickLabel', labels);
    xtickangle(45); grid on;
    
    % EXP3 selections
    subplot(3,1,3);
    bar(stats.mean_n_selected_exp3);
    title('Average Arm Selections - EXP3');
    set(gca, 'XTick', 1:length(labels), 'XTickLabel', labels);
    xtickangle(45); grid on;
    
    % 4. Time Performance
    figure('Name', 'Time Performance', 'Position', [100 100 1200 800]);
    
    % Average execution time
    subplot(2,2,1);
    times = [stats.timing.mean_times.ucb, stats.timing.mean_times.eps, stats.timing.mean_times.exp3];
    stds = [stats.timing.std_times.ucb, stats.timing.std_times.eps, stats.timing.std_times.exp3];
    bar(times);
    hold on;
    errorbar(1:3, times, stds, 'k.');
    set(gca, 'XTickLabel', {'UCB', '\epsilon-greedy', 'EXP3'});
    title('Average Execution Time');
    ylabel('Time (seconds)');
    grid on;
    
    % Execution time distribution
    subplot(2,2,2);
    boxplot([stats.timing.ucb_times, stats.timing.eps_times, stats.timing.exp3_times], ...
        'Labels', {'UCB', '\epsilon-greedy', 'EXP3'});
    title('Execution Time Distribution');
    ylabel('Time (seconds)');
    grid on;
    
    % Add summary text
    subplot(2,2,3);
    axis off;
    text(0.1, 0.8, sprintf('Total Computation Time: %.2fs', sum(times)));
    text(0.1, 0.6, sprintf('Fastest Algorithm: %.3fs', min(times)));
    text(0.1, 0.4, sprintf('Slowest Algorithm: %.3fs', max(times)));
end

function plot_with_bands(t, mean_data, std_data, color, label)
    plot(t, mean_data, color, 'LineWidth', 1.5, 'DisplayName', label);
    fill([t, fliplr(t)], [mean_data - std_data, fliplr(mean_data + std_data)], ...
        color, 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
end