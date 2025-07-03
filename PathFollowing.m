%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Path Following Matlab Function %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x1dot,x2dot,y1dot,y2dot,z1dot,z2dot,...
    Obstacle1Pos,Obstacle2Pos,Obstacle3Pos,Obstacle4Pos] = ...
    PathFollowing(xde,yde,zde,xdotde,ydotde,zdotde,...
    x1,x2,y1,y2,z1,z2,t)
% x1 = x (position)
% x2 = Vx (velocity = xdot)
% (xde,yde,zde) = desired path from path palanning block
% ========================= Constant Parameters ===========================
C1L = 5;                  % Kp Const. for position
C2L = 5;                  % Kd Const. for Velocity
KoS = 500;               % Static Obstacle Avoidance Const.
KoD = 350;               % Dynamic Obstacle Avoidance Const.

% ======================= Obstacle Definitions ============================
% 0-0-0-0-0-0-0-0-0-0-0-0-0-0 Static Obstacle 0-0-0-0-0-0-0-0-0-0-0-0-0-0-0
% ------------------------ cylindrical Obstacles --------------------------
xo1 = 20; yo1 = 5;  zo1 = 5;     Ro1 = 3.5; Ra1 = 10;
xo2 = 20; yo2 = -5; zo2 = 5;     Ro2 = 3.5; Ra2 = 10;
% ------------------------ Spherical Obstacle -----------------------------
xo3 = 40; yo3 = 0;  zo3 = 4.5;   Ro3 = 2; Ra3 = 10;
% 0-0-0-0-0-0-0-0-0-0-0-0-0-0 Dynamic Obstacle 0-0-0-0-0-0-0-0--0-0-0-0-0-0
xpos4 = 55; ypos4 = 0; zpos4 = 4.5;
xo4 = xpos4 + sin(t); yo4 = ypos4*sin(0.4*t); zo4 = zpos4 + cos(t);
Ro4 = 1; Ra4 = 8;

% for sending to Workspace (with the To Workspace Block)
Obstacle1Pos = [xo1;yo1;zo1];
Obstacle2Pos = [xo2;yo2;zo2];
Obstacle3Pos = [xo3;yo3;zo3];
Obstacle4Pos = [xo4;yo4;zo4];

% ========================= Agent Position ================================
% Current positioد
agent_pos = [x1; y1; z1];
agent_vel = [x2; y2; z2];


% ~~~~~~~~ Check if the Quadrotor approaches any of the obstacles ~~~~~~~~~ 
% ------------------------- F_ObstacleAvoidance ---------------------------
F_OA = zeros(3,1);

obstacles = [xo1, xo2, xo3, xo4;
             yo1, yo2, yo3, yo4;
             zo1, zo2, zo3, zo4];
Ra_values = [Ra1, Ra2, Ra3, Ra4];
Ko_values = [KoS, KoS, KoS, KoD];

for obs_idx = 1:4
    obs_pos = obstacles(:, obs_idx);
    Ra = Ra_values(obs_idx);
    Ko = Ko_values(obs_idx);

    diff_vec = agent_pos - obs_pos;
        r = norm(diff_vec);

    % APF Obstacle Avoidance
        if r < Ra && r > 1e-6
            force_magnitude = Ko * (1/r - 1/Ra) / (r^2);
            % more intensity for obstacle 3 z-axis field to force climb above the obstacle     
            if obs_idx == 3
                force_direction = diff_vec / r;
                force_direction(3) = force_direction(3) * 5;
                force = force_magnitude * force_direction;
            else
                force = force_magnitude * (diff_vec / r);
            end
            F_OA(:,1) = F_OA(:,1) + force;
        end
end

% ~~~~~~~~~~~~~~~ U = F_PathFollowing + F_ObstacleAvoidance ~~~~~~~~~~~~~~~

x1dot = x2;
x2dot = - C1L*(x1 - xde) - C2L*(x2 - xdotde) + F_OA(1,1);

y1dot = y2;
y2dot = - C1L*(y1 - yde) - C2L*(y2 - ydotde) + F_OA(2,1);

z1dot = z2;
z2dot = - C1L*(z1 - zde) - C2L*(z2 - zdotde) + F_OA(3,1);

end
