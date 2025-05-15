function [regret, n_selected] = run_bandit_algorithm(algorithm, arms, config, max_true_reward)
    a = size(arms, 1);
    n_selected = zeros(1, a);
    regret = zeros(1, config.T);
    
    switch algorithm
        case 'ucb'
            % UCB implementation
            empirical_mean = zeros(1, a);
            ucb_scores = zeros(1, a);
            
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
                empirical_mean(arm_idx) = ((n_selected(arm_idx) - 1) * empirical_mean(arm_idx) + reward) / n_selected(arm_idx);
                ucb_scores(arm_idx) = empirical_mean(arm_idx) + sqrt((config.alpha * log(t)) / (2 * n_selected(arm_idx)));
                regret(t) = max_true_reward - reward;
            end
            
        case 'epsilon'
            % Epsilon-greedy implementation
            empirical_mean = zeros(1, a);
            
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
                empirical_mean(arm_idx) = ((n_selected(arm_idx) - 1) * empirical_mean(arm_idx) + reward) / n_selected(arm_idx);
                regret(t) = max_true_reward - reward;
            end
            
        case 'exp3'
            % EXP3 implementation
            weights = ones(1, a);
            
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
end