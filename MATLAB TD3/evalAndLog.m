%% =============== Evaluation and Logging After Training ==================

% Set the complexity level that was just trained
complexity_level = 1; % Change this to match your training

% Load the trained agent and reference path
agentFile = sprintf('td3_agent_complexity_%d.mat', complexity_level);
refFile = sprintf('reference_path_complexity_%d.csv', complexity_level);

if ~exist(agentFile, 'file') || ~exist(refFile, 'file')
    error('Required files not found. Make sure training completed successfully.');
end

% Load agent and reference path
load(agentFile, 'agent');
refPath = readtable(refFile);

% Create environment
env = PathFollowEnv(refPath);

% ===================== Run Evaluation Simulation =========================
fprintf('Running evaluation simulation...\n');

% Reset environment and get initial observation
obs = reset(env);
N = height(refPath);
followed_path = zeros(N, 6);  % [x, vx, y, vy, z, vz]
actions_taken = zeros(N, 3);  % [ax, ay, az]
rewards = zeros(N, 1);

for k = 1:N
    % Get action from trained agent (no exploration noise during evaluation)
    act = getAction(agent, obs);
    actions_taken(k, :) = act';
    
    % Take step in environment
    [obs, reward, done, info] = step(env, act);
    rewards(k) = reward;
    
    % Store the state (position and velocity)
    followed_path(k, :) = obs(1:6)';
    
    if done
        fprintf('Simulation ended early at step %d\n', k);
        followed_path = followed_path(1:k, :);
        actions_taken = actions_taken(1:k, :);
        rewards = rewards(1:k);
        break
    end
end

% ======================= Save Results ===================================
% Save followed path for Simulink import
csvwrite(sprintf('followed_path_TD3_complexity_%d.csv', complexity_level), followed_path);

% Save actions and rewards for analysis
csvwrite(sprintf('actions_TD3_complexity_%d.csv', complexity_level), actions_taken);
csvwrite(sprintf('rewards_TD3_complexity_%d.csv', complexity_level), rewards);

fprintf('Results saved:\n');
fprintf('  - followed_path_TD3_complexity_%d.csv\n', complexity_level);
fprintf('  - actions_TD3_complexity_%d.csv\n', complexity_level);
fprintf('  - rewards_TD3_complexity_%d.csv\n', complexity_level);

% ================= Generate Training Report =============================
generateTrainingReport(complexity_level);

% ================= Calculate Performance Metrics ========================
N_eval = size(followed_path, 1);
t_eval = refPath.time(1:N_eval);

% Extract positions
x_ref = refPath.xd(1:N_eval);
y_ref = refPath.yd(1:N_eval);  
z_ref = refPath.zd(1:N_eval);

x_follow = followed_path(:, 1);
y_follow = followed_path(:, 3);
z_follow = followed_path(:, 5);

% Calculate tracking errors
error_x = x_follow - x_ref;
error_y = y_follow - y_ref;
error_z = z_follow - z_ref;
error_total = sqrt(error_x.^2 + error_y.^2 + error_z.^2);

% Performance metrics
rmse_x = sqrt(mean(error_x.^2));
rmse_y = sqrt(mean(error_y.^2));
rmse_z = sqrt(mean(error_z.^2));
rmse_total = sqrt(mean(error_total.^2));
max_error = max(error_total);
final_reward = sum(rewards);

fprintf('\n=================== Performance Metrics ===================\n');
fprintf('RMSE X: %.4f m\n', rmse_x);
fprintf('RMSE Y: %.4f m\n', rmse_y); 
fprintf('RMSE Z: %.4f m\n', rmse_z);
fprintf('RMSE Total: %.4f m\n', rmse_total);
fprintf('Max Error: %.4f m\n', max_error);
fprintf('Total Reward: %.2f\n', final_reward);
fprintf('=========================================================\n');

% ================= Optional: Compare with multiple levels ================
% Uncomment the next line after training all complexity levels
% plotRewardTrends();