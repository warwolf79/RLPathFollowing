%% ====================== Initialize Environment =========================

% -------------------- Set desired complexity level -----------------------
% 1: vertical climb | 2: climb + cruise | 3: climb + cruise + helix
complexity_level = 1;  % <<< Change this to 2 or 3 for next curriculum stage

% ---------------------- Load path and environment ------------------------
try
    csvName = sprintf('reference_path_complexity_%d.csv', complexity_level);
    refPath = readtable(csvName);
    env = PathFollowEnv(refPath);
catch ME
    error('Failed to initialize environment: %s', ME.message);
end

% --------------- Resuming Feature if training interrupted ----------------
% Set this flag true if resuming from a saved agent
resumeTraining = false;  % <<< change to true when resuming
agentFile = sprintf('td3_agent_complexity_%d.mat', complexity_level);

% ----------------------- Load agent if resuming --------------------------
if resumeTraining
    fprintf("🔁 Resuming training from saved agent: %s\n", agentFile);
    if isfile(agentFile)
        load(agentFile, 'agent');
    else
        error("Saved agent not found. Cannot resume.");
    end
else
    % ----------------- Create Actor & Critic Networks -----------------
    try
        % Get environment info
        obsInfo = getObservationInfo(env);
        actInfo = getActionInfo(env);

        % Actor optimizer settings
        actorOpts = rlOptimizerOptions(...
            'LearnRate', 1e-4, ...
            'GradientThreshold', 1, ...
            'L2RegularizationFactor', 1e-3);

        % Create deterministic actor network
        actorNet = [
            featureInputLayer(obsInfo.Dimension(1), 'Name', 'input')
            fullyConnectedLayer(256)
            reluLayer
            fullyConnectedLayer(256)
            reluLayer
            fullyConnectedLayer(3)
            tanhLayer
            scalingLayer('Scale', 3, 'Name', 'action')
            ];

        actor = rlDeterministicActorRepresentation(...
            actorNet, obsInfo, actInfo, ...
            'Observation', {'input'}, 'Action', {'action'});

        % Critic optimizer settings
        criticOpts = rlOptimizerOptions(...
            'LearnRate', 1e-3, ...
            'GradientThreshold', 1, ...
            'L2RegularizationFactor', 1e-3);

        % Build critic network
        obsPath = featureInputLayer(obsInfo.Dimension(1), 'Name', 'obsIn');
        actPath = featureInputLayer(actInfo.Dimension(1), 'Name', 'actIn');
        commonPath = [
            concatenationLayer(1, 2, 'Name', 'concat')
            fullyConnectedLayer(256)
            reluLayer
            fullyConnectedLayer(256)
            reluLayer
            fullyConnectedLayer(1, 'Name', 'qVal')
            ];

        criticGraph = layerGraph(obsPath);
        criticGraph = addLayers(criticGraph, actPath);
        criticGraph = addLayers(criticGraph, commonPath);
        criticGraph = connectLayers(criticGraph, 'obsIn', 'concat/in1');
        criticGraph = connectLayers(criticGraph, 'actIn', 'concat/in2');

        critic1 = rlQValueRepresentation(...
            criticGraph, obsInfo, actInfo, ...
            'Observation', {'obsIn'}, 'Action', {'actIn'});
        critic2 = rlQValueRepresentation(...
            criticGraph, obsInfo, actInfo, ...
            'Observation', {'obsIn'}, 'Action', {'actIn'});

        % Configure TD3 Agent
        explorationModel = rl.option.GaussianActionNoise(...
            'Mean', 0, 'StandardDeviation', 0.45, ...
            'StandardDeviationDecayRate', 1e-6);
        targetPolicyNoise = rl.option.GaussianActionNoise(...
            'Mean', 0, 'StandardDeviation', 0.2, ...
            'LowerLimit', -0.5, 'UpperLimit', 0.5);

        agentOpts = rlTD3AgentOptions(...
            'SampleTime', env.Ts, ...
            'MiniBatchSize', 256, ...
            'ExperienceBufferLength', 5e6, ...
            'DiscountFactor', 0.99, ...
            'TargetSmoothFactor', 5e-3, ...
            'ExplorationModel', explorationModel, ...
            'TargetPolicySmoothModel', targetPolicyNoise, ...
            'ActorOptimizerOptions', actorOpts, ...
            'CriticOptimizerOptions', criticOpts);

        % Create agent
        agent = rlTD3Agent(actor, [critic1, critic2], agentOpts);
    catch ME
        error('Failed to create agent: %s', ME.message);
    end
end

%% ========================== Training Setup ==============================
maxSteps = height(refPath);

% Training options
trainOpts = rlTrainingOptions(...
    'MaxEpisodes', 500, ...
    'MaxStepsPerEpisode', maxSteps, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'StopTrainingCriteria', 'AverageReward', ...
    'StopTrainingValue', -1e-2);

% Create checkpoint directory
checkpointDir = 'checkpoints';
if ~exist(checkpointDir, 'dir')
    mkdir(checkpointDir);
end

% Start training
try
    fprintf('Starting training for complexity level %d...\n', complexity_level);
    trainingStats = trainWithCheckpoints(agent, env, trainOpts, complexity_level, checkpointDir);

    % Save final agent and training statistics
    save(agentFile, 'agent');
    save(sprintf('trainingStats_complexity_%d.mat', complexity_level), 'trainingStats');

    % Save episode rewards
    if isfield(trainingStats, 'EpisodeIndex') && isfield(trainingStats, 'EpisodeReward')
        episodeRewards = [trainingStats.EpisodeIndex', trainingStats.EpisodeReward'];
        writematrix(episodeRewards, sprintf('episode_rewards_complexity_%d.csv', complexity_level));
    end

    fprintf('Training completed for complexity level %d\n', complexity_level);
catch ME
    fprintf('Training failed: %s\n', ME.message);
    rethrow(ME);
end

%% Custom Training Function with Checkpoints
function trainingStats = trainWithCheckpoints(agent, env, trainOpts, complexity_level, checkpointDir)
    % Initialize training statistics
    episodeIndex = [];
    episodeReward = [];
    episodeSteps = [];
    
    maxEpisodes = trainOpts.MaxEpisodes;
    maxStepsPerEpisode = trainOpts.MaxStepsPerEpisode;
    
    try
        for episode = 1:maxEpisodes
            % Reset environment
            obs = reset(env);
            
            % Episode variables
            episodeRewardSum = 0;
            stepCount = 0;
            
            % Episode loop
            for step = 1:maxStepsPerEpisode
                % Get action from agent with validation
                try
                    action = getAction(agent, obs);
                    
                    % Ensure action is in the correct format
                    if iscell(action)
                        action = cell2mat(action);
                    end
                    if ~isnumeric(action)
                        error('Invalid action type from agent');
                    end
                    
                    % Reshape if needed
                    action = double(reshape(action, [3,1]));
                    
                catch ME
                    fprintf('Error getting action: %s\n', ME.message);
                    rethrow(ME);
                end
                
                % Take step in environment
                try
                    [nextObs, reward, done, info] = env.step(action);
                catch ME
                    fprintf('Error in environment step: %s\nAction was: %s\n', ...
                        ME.message, mat2str(action));
                    rethrow(ME);
                end
                
                % Store experience using experience method
                try
                    % Create experience
                    experience = struct(...
                        'Observation', obs, ...
                        'Action', action, ...
                        'Reward', reward, ...
                        'NextObservation', nextObs, ...
                        'IsDone', done);
                    
                    % Add experience to agent's buffer
                    agent.remember(experience);
                    
                catch ME
                    fprintf('Error storing experience: %s\n', ME.message);
                    rethrow(ME);
                end
                
                % Learn if buffer has enough samples
                if agent.ExperienceBuffer.Length >= agent.AgentOptions.MiniBatchSize
                    try
                        agent = learn(agent);
                    catch ME
                        fprintf('Error during learning: %s\n', ME.message);
                        rethrow(ME);
                    end
                end
                
                % Update for next iteration
                obs = nextObs;
                episodeRewardSum = episodeRewardSum + reward;
                stepCount = stepCount + 1;
                
                if done
                    break;
                end
            end
            
            % Store episode statistics
            episodeIndex(end+1) = episode;
            episodeReward(end+1) = episodeRewardSum;
            episodeSteps(end+1) = stepCount;
            
            % Display progress
            if mod(episode, 10) == 0
                avgReward = mean(episodeReward(max(1, end-49):end));
                fprintf('Episode %d: Reward = %.3f, Avg Reward (last 50) = %.3f, Steps = %d\n', ...
                    episode, episodeRewardSum, avgReward, stepCount);
            end
            
            % Save checkpoint
            if mod(episode, 100) == 0
                checkpointFile = fullfile(checkpointDir, ...
                    sprintf('agent_complexity_%d_episode_%d.mat', complexity_level, episode));
                save(checkpointFile, 'agent');
                fprintf('Checkpoint saved: %s\n', checkpointFile);
            end
            
            % Check stopping criteria
            if length(episodeReward) >= 50
                avgReward = mean(episodeReward(end-49:end));
                if avgReward >= trainOpts.StopTrainingValue
                    fprintf('Training stopped: Average reward target reached (%.3f)\n', avgReward);
                    break;
                end
            end
        end
    catch ME
        fprintf('Error during training: %s\n', ME.message);
        rethrow(ME);
    end
    
    % Create training statistics structure
    trainingStats.EpisodeIndex = episodeIndex;
    trainingStats.EpisodeReward = episodeReward;
    trainingStats.EpisodeSteps = episodeSteps;
    
    % Plot training progress
    try
        figure('Name', 'Training Progress', 'NumberTitle', 'off');
        subplot(2,1,1);
        plot(episodeIndex, episodeReward);
        xlabel('Episode');
        ylabel('Episode Reward');
        title('Episode Reward vs Episode');
        grid on;
        
        subplot(2,1,2);
        windowSize = min(50, length(episodeReward));
        if length(episodeReward) >= windowSize
            movingAvg = movmean(episodeReward, windowSize);
            plot(episodeIndex, movingAvg);
            xlabel('Episode');
            ylabel('Average Reward');
            title(sprintf('Moving Average Reward (%d episodes)', windowSize));
            grid on;
        end
    catch ME
        warning('Failed to create training plots: %s', ME.message);
    end
end