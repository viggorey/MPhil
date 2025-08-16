% Visualization of 3D Trajectories from Exported Excel Dataset
% Define the base path and file name
base_path = '/Users/viggorey/Desktop/PhD/Cambridge/Macaranga/3D transformation/5. Datasets/3D data/Large branch/';
file_prefix = '22U10'; % Set this to the prefix used during export
input_file = fullfile(base_path, [file_prefix, '.xlsx']); % Excel file path

% Get the list of valid sheets in the Excel file
sheet_names = sheetnames(input_file); % Retrieves all valid sheet names

% Initialize storage for trajectory data
all_trajectories = cell(length(sheet_names), 1);


% Loop through each sheet to read the data
for i = 1:length(sheet_names)
    % Read the data from the current sheet
    data = readtable(input_file, 'Sheet', sheet_names{i}, 'VariableNamingRule', 'preserve');

    % Store the data for visualization
    all_trajectories{i} = data;
end

% Compute the number of time frames from the first trajectory
num_time_frames = size(all_trajectories{1}, 1);

% Create the 17th track as the midpoint between Track 2 and Track 3
midpoint_data = table();
midpoint_data.X = (all_trajectories{2}.X + all_trajectories{3}.X) / 2;
midpoint_data.Y = (all_trajectories{2}.Y + all_trajectories{3}.Y) / 2;
midpoint_data.Z = (all_trajectories{2}.Z + all_trajectories{3}.Z) / 2;
midpoint_data.Residual = NaN(num_time_frames, 1); % Residuals are not applicable
midpoint_data.CamerasUsed = repmat({'Midpoint'}, num_time_frames, 1); % Label as "Midpoint"

% Add the 17th track to the all_trajectories array
all_trajectories{end + 1} = midpoint_data;

% Update sheet names to include the 17th track
sheet_names{end + 1} = 'Track 17 (Midpoint)';

% Define the pairs of tracks to connect with dynamic lines
track_pairs = [1, 2; 2, 3; 3, 4; 5, 8; 6, 9; 7, 10; 11, 14; 12, 15; 13, 16];

% Additional connections between Track 17 and specified tracks
midpoint_connections = [17, 5; 17, 6; 17, 7; 17, 11; 17, 12; 17, 13];

% Combine all track pairs
all_track_pairs = [track_pairs; midpoint_connections];

% Create a figure for the animation
figure;
hold on;
axis equal;
grid on;

% Set up the 3D plot limits (you can adjust based on your data range)
xlim([-10, 10]); % Adjust based on your data
ylim([-10, 10]); % Adjust based on your data
zlim([-10, 10]); % Adjust based on your data

% Add labels and title
xlabel('X');
ylabel('Y');
zlabel('Z');
title('3D Movement Animation with Midpoint Track');

% Create an array of plot handles for the points
plot_handles = gobjects(length(all_trajectories), 1);

% Initialize each point in the plot
for i = 1:length(all_trajectories)
    if i == 17 % Special styling for the midpoint track
        plot_handles(i) = plot3(NaN, NaN, NaN, 'ko', 'MarkerSize', 4); % Very small black dot
    else
        plot_handles(i) = plot3(NaN, NaN, NaN, 'ko', 'MarkerSize', 8); % Regular black dot
    end
end

% Create an array of line handles for the track pairs
line_handles = gobjects(size(all_track_pairs, 1), 1);

% Initialize each dynamic line
for i = 1:size(all_track_pairs, 1)
    line_handles(i) = plot3(NaN, NaN, NaN, 'k-', 'LineWidth', 2); % Black lines
end

% Animation loop
while true % Infinite loop for continuous animation
    for t = 1:num_time_frames
        % Update each track's position at the current time frame
        for i = 1:length(all_trajectories)
            % Extract the current X, Y, Z coordinates
            X = all_trajectories{i}.X(t);
            Y = all_trajectories{i}.Y(t);
            Z = all_trajectories{i}.Z(t);

            % Update the plot handle's data
            set(plot_handles(i), 'XData', X, 'YData', Y, 'ZData', Z);
        end

        % Update the dynamic lines for each track pair
        for i = 1:size(all_track_pairs, 1)
            track1 = all_track_pairs(i, 1); % First track in the pair
            track2 = all_track_pairs(i, 2); % Second track in the pair

            X_line = [all_trajectories{track1}.X(t), all_trajectories{track2}.X(t)];
            Y_line = [all_trajectories{track1}.Y(t), all_trajectories{track2}.Y(t)];
            Z_line = [all_trajectories{track1}.Z(t), all_trajectories{track2}.Z(t)];

            % Update the line handle's data
            set(line_handles(i), 'XData', X_line, 'YData', Y_line, 'ZData', Z_line);
        end

        % Pause to create slower animation effect
        pause(0.2); % Adjust the pause duration to control animation speed
    end
end