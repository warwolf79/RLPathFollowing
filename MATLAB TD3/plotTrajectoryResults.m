function plotTrajectoryResults(refPath, followed_path)
    t = refPath.time;
    N = min(size(followed_path,1), height(refPath));

    % Trim to match size
    t = t(1:N);
    xd = refPath.xd(1:N); yd = refPath.yd(1:N); zd = refPath.zd(1:N);
    x = followed_path(1:N,1); y = followed_path(1:N,3); z = followed_path(1:N,5);

    % ========== 3D Trajectory ==========
    figure('Name','3D Path Comparison','NumberTitle','off');
    plot3(xd, yd, zd, 'b--', 'LineWidth', 2); hold on;
    plot3(x, y, z, 'r', 'LineWidth', 2);
    legend('Reference', 'TD3 Followed');
    xlabel('X'); ylabel('Y'); zlabel('Z');
    grid on; axis equal;
    title('3D Path Comparison');

    % ========== Position Errors ==========
    err_x = x - xd; err_y = y - yd; err_z = z - zd;

    figure('Name','Position Errors','NumberTitle','off');
    subplot(3,1,1); plot(t, err_x, 'r'); ylabel('X Error (m)'); grid on;
    subplot(3,1,2); plot(t, err_y, 'g'); ylabel('Y Error (m)'); grid on;
    subplot(3,1,3); plot(t, err_z, 'b'); ylabel('Z Error (m)'); xlabel('Time (s)'); grid on;
    sgtitle('Tracking Errors');

    % ========== Velocities ==========
    vx = followed_path(1:N,2);
    vy = followed_path(1:N,4);
    vz = followed_path(1:N,6);

    figure('Name','Agent Velocity','NumberTitle','off');
    subplot(3,1,1); plot(t, vx, 'r'); ylabel('Vx (m/s)'); grid on;
    subplot(3,1,2); plot(t, vy, 'g'); ylabel('Vy (m/s)'); grid on;
    subplot(3,1,3); plot(t, vz, 'b'); ylabel('Vz (m/s)'); xlabel('Time (s)'); grid on;
    sgtitle('Agent Velocities');

    % ========== Acceleration Magnitude ==========
    % Optional: Plot actions if logged
end
