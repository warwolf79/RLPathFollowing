function generateTrainingReport(complexity_level)
    % === Load reference path and agent ===
    refFile = sprintf('reference_path_complexity_%d.csv', complexity_level);
    agentFile = sprintf('td3_agent_complexity_%d.mat', complexity_level);
    statsFile = sprintf('trainingStats_complexity_%d.mat', complexity_level);
    
    if ~isfile(agentFile) || ~isfile(refFile)
        error('Required files not found.');
    end
    
    refPath = readtable(refFile);
    load(agentFile, 'agent');

    % === Re-create environment ===
    env = PathFollowEnv(refPath);
    env.reset();
    obs = env.State;

    % === Run Evaluation ===
    N = height(refPath);
    followed_path = zeros(N, 6); % [x, vx, y, vy, z, vz]
    
    for k = 1:N
        act = getAction(agent, obs);
        [obs, ~, done, ~] = step(env, act);
        followed_path(k,:) = obs(1:6)';
        if done
            followed_path = followed_path(1:k,:);
            break
        end
    end

    % === Save output CSV ===
    csvwrite(sprintf('followed_path_TD3_complexity_%d.csv', complexity_level), followed_path);

    % === Plot Trajectory and Errors ===
    plotTrajectoryResults(refPath, followed_path);

    % === Plot Reward Trend ===
    if isfile(statsFile)
        load(statsFile, 'trainingStats');
        figure('Name', 'Reward Trend', 'NumberTitle', 'off');
        plot(trainingStats.EpisodeIndex, trainingStats.EpisodeReward, 'LineWidth', 2);
        xlabel('Episode'); ylabel('Reward');
        title(sprintf('TD3 Reward Trend - Complexity %d', complexity_level));
        grid on;
    end
end
