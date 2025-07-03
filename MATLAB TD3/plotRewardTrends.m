function plotRewardTrends()
    figure('Name','Reward Trends by Complexity','NumberTitle','off'); hold on;
    colors = {'b','g','r'};
    
    for level = 1:3
        file = sprintf('trainingStats_complexity_%d.mat', level);
        if exist(file, 'file')
            load(file, 'trainingStats');
            plot(trainingStats.EpisodeIndex, trainingStats.EpisodeReward, ...
                'Color', colors{level}, 'DisplayName', sprintf('Complexity %d', level), ...
                'LineWidth', 1.5);
        end
    end

    xlabel('Episode');
    ylabel('Total Reward');
    title('TD3 Agent Reward Trends Across Curriculum');
    legend('show'); grid on;
end
