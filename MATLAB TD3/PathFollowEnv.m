classdef PathFollowEnv < rl.env.MATLABEnvironment
    properties
        % Environment properties (step size, current index)
        Ts = 0.01
        CurrentStep = 1
        MaxSteps = 13500
        PathData
        State = zeros(16,1)
    end

    properties(Access = protected)
        IsDone = false;
    end

    methods
        function this = PathFollowEnv(pathTable)
            % Define observation and action spaces
            ObservationInfo = rlNumericSpec([16 1], ...
                'LowerLimit', -inf, 'UpperLimit', inf);
            ActionInfo = rlNumericSpec([3 1], ...
                'LowerLimit', -3, 'UpperLimit', 3); % 3D acceleration bounds
            
            % Call superclass constructor
            this = this@rl.env.MATLABEnvironment(ObservationInfo, ActionInfo);

            % Store path data
            this.PathData = pathTable;
            this.MaxSteps = height(pathTable);
            
            % Initialize state
            this.reset();
        end

        function [nextObs, reward, done, info] = step(this, action)
    % Input validation for action
    if ~isnumeric(action)
        error('Action must be numeric');
    end
    
    % Ensure action is a 3x1 column vector
    if isa(action, 'cell')
        action = cell2mat(action);
    elseif isa(action, 'gpuArray')
        action = gather(action);
    end
    
    % Convert to double if needed
    action = double(action);
    
    % Reshape if needed
    if ~isequal(size(action), [3,1])
        action = reshape(action, [3,1]);
    end
    
    % Ensure action is column vector
    action = action(:);
    
    % Extract current state
    s = this.State;
    
    % Check if we've reached the end
    if this.CurrentStep > height(this.PathData)
        nextObs = this.State;
        reward = -1000; % Large penalty for going beyond path
        done = true;
        this.IsDone = true;
        info = struct('termination_reason', 'path_end');
        return;
    end

    % Extract desired path point
    ref = this.PathData(this.CurrentStep, :);
    xde = ref.xd; 
    yde = ref.yd; 
    zde = ref.zd;
    xdotde = ref.xdot; 
    ydotde = ref.ydot; 
    zdotde = ref.zdot;

    % Current state components
    pos = [s(1); s(3); s(5)];  % [x, y, z]
    vel = [s(2); s(4); s(6)];  % [vx, vy, vz]
    
    % Apply acceleration limits componentwise to avoid matrix issues
    acc = zeros(3,1);
    for i = 1:3
        acc(i) = max(-3, min(3, action(i)));
    end
    
    % Update dynamics with validated dt
    dt = double(this.Ts);
    vel = vel + acc * dt;
    pos = pos + vel * dt;

    % Get obstacle distances
    obsDists = this.getObstacleDistances(pos);

    % Create next observation (ensure all components are double)
    nextObs = double([
        pos(1); vel(1);     % x, vx
        pos(2); vel(2);     % y, vy  
        pos(3); vel(3);     % z, vz
        xde; xdotde;        % desired x, vx
        yde; ydotde;        % desired y, vy
        zde; zdotde;        % desired z, vz
        obsDists            % 4 obstacle distances
    ]);

    % Save state
    this.State = nextObs;
    this.CurrentStep = this.CurrentStep + 1;

    % Calculate reward components
    pos_error = norm(pos - [xde; yde; zde]);
    vel_error = norm(vel - [xdotde; ydotde; zdotde]);
    
    % Reward components
    tracking_reward = -10 * pos_error^2 - vel_error^2;
    control_penalty = -0.01 * norm(acc)^2;
    
    % Obstacle penalty with safety margin
    obstacle_penalty = 0;
    min_safe_distance = 5.0; % meters
    collision_distance = 1.5; % collision threshold
    
    for i = 1:length(obsDists)
        if obsDists(i) < collision_distance
            obstacle_penalty = obstacle_penalty - 1000; % Large collision penalty
        elseif obsDists(i) < min_safe_distance
            obstacle_penalty = obstacle_penalty - 100 * (min_safe_distance - obsDists(i))^2;
        end
    end
    
    % Progress reward
    progress_reward = 0;
    if this.CurrentStep > 1
        progress_reward = 1;
    end
    
    % Compute total reward
    reward = double(tracking_reward + control_penalty + obstacle_penalty + progress_reward);

    % Termination conditions
    collision = any(obsDists < collision_distance);
    too_far = pos_error > 20;
    out_of_bounds = pos(3) < -1 || abs(pos(1)) > 100 || abs(pos(2)) > 50;
    
    done = this.CurrentStep > height(this.PathData) || collision || too_far || out_of_bounds;
    this.IsDone = done;
    
    % Create info structure
    info = struct('pos_error', pos_error, 'vel_error', vel_error, ...
                 'min_obstacle_dist', min(obsDists), ...
                 'collision', collision, 'too_far', too_far, ...
                 'out_of_bounds', out_of_bounds);
end

        function initialObs = reset(this)
            % Reset environment to initial conditions
            this.CurrentStep = 1;
            this.IsDone = false;
            
            % Get initial reference point
            if height(this.PathData) > 0
                ref = this.PathData(1, :);
                xde = ref.xd; yde = ref.yd; zde = ref.zd;
                xdotde = ref.xdot; ydotde = ref.ydot; zdotde = ref.zdot;
            else
                xde = 0; yde = 0; zde = 0;
                xdotde = 0; ydotde = 0; zdotde = 0;
            end
            
            % Initialize state near the starting point with small random noise
            pos = [xde; yde; zde] + 0.05 * randn(3,1); % Reduced noise
            vel = [xdotde; ydotde; zdotde] + 0.01 * randn(3,1); % Small velocity noise
            
            % Get obstacle distances
            obsDists = this.getObstacleDistances(pos);
            
            % Create initial observation
            this.State = [
                pos(1); vel(1);     % x, vx
                pos(2); vel(2);     % y, vy
                pos(3); vel(3);     % z, vz  
                xde; xdotde;        % desired x, vx
                yde; ydotde;        % desired y, vy
                zde; zdotde;        % desired z, vz
                obsDists            % 4 obstacle distances
            ];
            
            initialObs = this.State;
        end

        function dists = getObstacleDistances(this, pos)
            % Calculate distances to obstacles based on current time
            t = (this.CurrentStep - 1) * this.Ts;
            
            % Static obstacles (from your MATLAB function)
            obs_static = [
                20, 5, 5;      % Cylindrical obstacle 1
                20, -5, 5;     % Cylindrical obstacle 2  
                40, 0, 4.5;    % Spherical obstacle 3
            ];
            
            % Dynamic obstacle 4 (from your MATLAB function)
            xo4 = 55 + sin(t);
            yo4 = 0 * sin(0.4 * t);
            zo4 = 4.5 + cos(t);
            obs_dynamic = [xo4, yo4, zo4];
            
            % Combine all obstacles
            obstacles = [obs_static; obs_dynamic];
            
            % Calculate distances
            dists = zeros(4, 1);
            for i = 1:4
                dists(i) = norm(pos - obstacles(i, :)');
            end
        end
        
        function enforceResetMethod(this)
            % Ensure reset is called when environment is used with RL agents
            reset(this);
        end
    end
end