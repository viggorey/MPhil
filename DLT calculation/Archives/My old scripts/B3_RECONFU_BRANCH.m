% Define: 
base_path = '/Users/viggorey/Desktop/PhD/Cambridge/Macaranga/3D transformation/6. Branch'; % Base path
camera_files = {'TOP.xlsx', 'LEFT.xlsx', 'RIGHT.xlsx'}; % Camera file names
camera_names = {'Top', 'Left', 'Right'}; % Camera names for reference
num_cameras = length(camera_files); % Number of cameras
num_points = 19; % Number of tracking points (28 tracks)

% Load DLT coefficients for each camera
A = [
    88.2106, 93.4303, 78.7655; % Camera 1 - first coefficient for all cameras
    -6.6498, -14.1814, -3.1569; % Camera 2
    37.76, -2.7843, -45.5904; 
    608.6661, 378.2291, 374.2239; 
    -2.8612, 1.9343, -0.1604; 
    140.3325, 131.9784, 134.1519; 
    0.6305, 1.0066, 7.2696; 
    528.5439, 500.3596, 502.5081; 
    0.0011, 0.001, 0.0015; 
    -0.013, -0.0208, -0.0149; 
    -0.0017, -0.0006, 0.0019
];

% Initialize cell arrays to store L matrices
L_matrices = cell(1, num_points);

% Read and process data from all cameras
camera_data = cell(1, num_cameras);

for cam = 1:num_cameras
    % Read the camera data
    file_path = fullfile(base_path, camera_files{cam});
    % Read the table and inspect variable names
    data = readtable(file_path, 'VariableNamingRule', 'preserve'); % Preserve original column headers
    disp(data.Properties.VariableNames); % Display column names to identify the issue

    % Extract X and Y coordinates for all tracks
    x_data = data{strcmp(data.('Coordinate Type'), 'X'), 3:end}; % Adjust column name here
    y_data = data{strcmp(data.('Coordinate Type'), 'Y'), 3:end}; % Adjust column name here
    
    % Combine X and Y into a single matrix for this camera
    camera_data{cam} = [x_data; y_data];
end

% Construct L matrices for each tracking point
for point = 1:num_points
    % Initialize L matrix for this tracking point
    num_frames = size(camera_data{1}, 2); % Number of frames (columns)
    L = NaN(num_frames, 2 * num_cameras); % Rows: frames, Columns: 2 (X, Y) per camera

    for cam = 1:num_cameras
        % Extract X and Y data for this point from the current camera
        X = camera_data{cam}(point, :); % X coordinates for this point
        Y = camera_data{cam}(num_points + point, :); % Y coordinates for this point

        % Fill in the columns for this camera in the L matrix
        L(:, (cam - 1) * 2 + 1) = X';
        L(:, (cam - 1) * 2 + 2) = Y';
    end

    % Store the constructed L matrix for this tracking point
    L_matrices{point} = L;
end

disp('L matrices successfully constructed!');

function [H_all] = RECONFU_parallel(A, L_matrices)
    % Number of points
    num_points = length(L_matrices);
    % Preallocate outputs
    H_all = cell(1, num_points);

    for p = 1:num_points
        % Extract L matrix
        L = L_matrices{p};
        n = size(A, 2); % Number of cameras

        if 2 * n ~= size(L, 2)
            disp('Mismatch in number of cameras. Skipping point.');
            H_all{p} = NaN;
            continue;
        end

        % Initialize H
        H = NaN(size(L, 1), 4); % Columns: X, Y, Z, Residual
        camsused_cell = cell(size(L, 1), 1);

        for k = 1:size(L, 1) % Iterate over frames
            q = 0; L1 = []; L2 = [];
            valid_cameras = true(1, n); % Track valid cameras for this frame

            for i = 1:n
                x = L(k, 2 * i - 1); % X coordinate
                y = L(k, 2 * i);     % Y coordinate

                % Mark camera as invalid if X or Y is 0
                if x == 0 || y == 0
                    valid_cameras(i) = false;
                else
                    % Include this camera's data in L1 and L2
                    q = q + 1;
                    L1([q * 2 - 1:q * 2], :) = [A(1, i) - x * A(9, i), A(2, i) - x * A(10, i), A(3, i) - x * A(11, i); ...
                                                A(5, i) - y * A(9, i), A(6, i) - y * A(10, i), A(7, i) - y * A(11, i)];
                    L2([q * 2 - 1:q * 2], :) = [x - A(4, i); y - A(8, i)];
                end
            end

            if (size(L2, 1) / 2) > 1 % Enough valid cameras for reconstruction
                g = L1 \ L2; % Compute 3D coordinates
                h = L1 * g; % Recalculate 2D projections
                DOF = size(L2, 1) - 3; % Degrees of freedom
                avgres = sqrt(sum((L2 - h).^2) / DOF); % Residuals
            else
                g = [NaN; NaN; NaN]; 
                avgres = NaN;
            end

            % Generate camera labels for valid cameras
            camera_labels = {'T', 'L', 'R'}; % Top, Left, Right
            camsused = strjoin(camera_labels(valid_cameras), ''); % Concatenate labels of valid cameras

            % Store results
            H(k, :) = [g', avgres]; % X, Y, Z, Residual
            camsused_cell{k} = camsused;
        end

        H_all{p} = [num2cell(H), camsused_cell];
    end
end

% Perform 3D Reconstruction
H_all = RECONFU_parallel(A, L_matrices);

% Export Results to Excel
output_file = fullfile(base_path, '3D_Coordinates.xlsx');
writer = table();

for point = 1:num_points
    % Extract 3D data for this point
    H = H_all{point};
    if isempty(H)
        continue;
    end

    % Create a table
    point_data = table((1:size(H, 1))', H(:, 1), H(:, 2), H(:, 3), H(:, 4), ...
        'VariableNames', {'Frame', 'X', 'Y', 'Z', 'Residual'});
    point_data.Track = repmat(point, size(H, 1), 1); % Add track ID
    writer = [writer; point_data]; % Append to the full dataset
end

writetable(writer, output_file);
disp(['3D coordinates saved to ', output_file]);