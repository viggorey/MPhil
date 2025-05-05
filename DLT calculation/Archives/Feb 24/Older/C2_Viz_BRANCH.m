% Load the dataset
% If you have the dataset saved as a file, replace 'data' with the file reading code
data = readtable('/Users/viggorey/Desktop/PhD/Cambridge/Macaranga/3D transformation/6. Branch/3D_Coordinates.xlsx');

% Extract columns for visualization
X = data.X; % X coordinates
Y = data.Y; % Y coordinates
Z = data.Z; % Z coordinates
Tracks = data.Track; % Track identifiers

% Ensure Tracks is treated as an array
if istable(Tracks)
    Tracks = table2array(Tracks); % Convert table column to array
end

% Unique tracks for coloring
unique_tracks = unique(Tracks);
num_tracks = numel(unique_tracks);

% Create a colormap with a distinct color for each track
cmap = lines(num_tracks);

% Plot the 3D points
figure;
hold on;
for i = 1:num_tracks
    % Extract data for the current track
    track_indices = Tracks == unique_tracks(i);
    track_X = X(track_indices);
    track_Y = Y(track_indices);
    track_Z = Z(track_indices);
    
    % Plot the track
    plot3(track_X, track_Y, track_Z, 'o-', 'Color', cmap(i, :), 'DisplayName', ['Track ', num2str(unique_tracks(i))]);
end

% Add labels, title, and legend
xlabel('X');
ylabel('Y');
zlabel('Z');
title('3D Visualization of Points by Track');
legend('show');
grid on;
hold off;

% Set aspect ratio for better visualization
axis equal;



% Delaunay triangulation
tri = delaunay(X, Y);

% Plot the surface
figure;
trisurf(tri, X, Y, Z, 'EdgeColor', 'none');
colormap jet; % Add color based on Z values
colorbar; % Add a color bar to show the Z scale
xlabel('X');
ylabel('Y');
zlabel('Z');
title('3D Surface Reconstruction');
axis equal;
view(3); % Set 3D view
grid on;