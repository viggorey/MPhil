% Define: 
base_path = '/Users/viggorey/Desktop/PhD/Cambridge/Macaranga/3D transformation/5. Datasets/2D data/Large branch/'; % Base path
file_prefix = '21D2'; % Common file prefix
camera_suffixes = {'L', 'T', 'R', 'F'}; % Camera suffixes (Top, Left, Right, Front)


A = [
    83.5273, 92.518, 81.1897, 87.1662;
    1.4885, -3.1049, 3.8105, 1.3689;
    48.5334, 7.2765, -38.6041, 17.7288;
    692.1982, 757.5879, 652.8448, 663.2397;
    -0.0698, 1.9538, -1.8253, 3.4876;
    149.2815, 146.887, 142.6634, 127.964;
    1.106, 0.0965, 4.574, -38.4277;
    302.1854, 317.1885, 405.0939, 354.242;
    0.0031, 0.0008, -0.0011, -0.0018;
    0.0005, 0.0003, -0.0009, -0.004;
    -0.0002, -0.0015, -0.001, 0.0013
];


% Construct the full file paths dynamically
camera_files = strcat(base_path, file_prefix, camera_suffixes, '.xlsx');camera_names = {'Left', 'Top',  'Right', 'Front'};
num_cameras = length(camera_files);
num_points = 16; % Number of tracking points

% Initialize cell arrays to store L matrices
L_matrices = cell(1, num_points);

% Read and process data from all cameras
camera_data = cell(1, num_cameras);

for cam = 1:num_cameras
    % Read the camera data and transpose to match the expected format
    data = readmatrix(camera_files{cam})'; % Transpose data

    % Replace 0 values with NaN
    data(data == 0) = NaN;

    % Store the processed data
    camera_data{cam} = data;
end

% Verify the dimensions
for cam = 1:num_cameras
    disp(['Camera ', camera_names{cam}, ': ', num2str(size(camera_data{cam}, 1)), ...
          ' time frames, ', num2str(size(camera_data{cam}, 2)), ' points tracked']);
end

% Construct L matrices for each tracking point
for point = 1:num_points
    % Initialize L matrix for this tracking point
    time_frames = size(camera_data{1}, 1); % Rows now represent time frames
    L = NaN(time_frames, 2 * num_cameras); % Rows: time frames, Columns: 2 (X, Y) per camera

    for cam = 1:num_cameras
        % Extract X and Y data for this point from the current camera
        X = camera_data{cam}(:, point * 2 - 1); % Adjusted indexing for transposed data
        Y = camera_data{cam}(:, point * 2);     % Adjusted indexing for transposed data

        % Fill in the columns for this camera in the L matrix
        L(:, (cam - 1) * 2 + 1) = X;
        L(:, (cam - 1) * 2 + 2) = Y;
    end

    % Store the constructed L matrix for this tracking point
    L_matrices{point} = L;
end

disp('L matrices successfully constructed!');


function [H_all] = RECONFU_parallel(A, L_matrices)

    % Get the number of points to process
    num_points = length(L_matrices);

    % Preallocate a cell array for outputs
    H_all = cell(1, num_points);

    % Start parallel processing
    parfor p = 1:num_points
        % Extract the L matrix for the current point
        L = L_matrices{p};

        % Call the original RECONFU logic for this point
        n = size(A, 2);

        % Check if the numbers of cameras in A and L agree
        if 2 * n ~= size(L, 2)
            disp('The # of cameras given in A and L do not agree');
            disp('Skipping point'); 
            H_all{p} = NaN;
            continue;
        end

        % Initialize H as a numeric matrix for the first four columns
        H = NaN(size(L, 1), 4); % Columns: X, Y, Z, Residual
        camsused_cell = cell(size(L, 1), 1); % Separate cell array for camera strings

        % Reconstruction logic for each time frame
        for k = 1:size(L, 1)  % Number of time points
            q = 0; 
            L1 = []; 
            L2 = [];  % Initialize L1, L2, and camera counter
            for i = 1:n  % Number of cameras
                x = L(k, 2 * i - 1); 
                y = L(k, 2 * i);  % Camera coordinates
                if ~(isnan(x) | isnan(y))  % Valid camera data
                    q = q + 1;
                    L1([q * 2 - 1:q * 2], :) = [A(1, i) - x * A(9, i), A(2, i) - x * A(10, i), A(3, i) - x * A(11, i); ...
                                                A(5, i) - y * A(9, i), A(6, i) - y * A(10, i), A(7, i) - y * A(11, i)];
                    L2([q * 2 - 1:q * 2], :) = [x - A(4, i); y - A(8, i)];
                end
            end

            if (size(L2, 1) / 2) > 1  % Enough cameras for reconstruction
                g = L1 \ L2;
                h = L1 * g;
                DOF = (size(L2, 1) - 3);  % Degrees of freedom
                avgres = sqrt(sum([L2 - h].^2) / DOF);
            else
                g = [NaN; NaN; NaN]; 
                avgres = NaN;
            end

            % Define camera labels
            camera_labels = {'L', 'T', 'R', 'F'}; % Front, Top, Left, Right

            % Identify cameras used
            valid_cameras = ~isnan(L(k, 1:2:end)); % Check validity of X data for each camera
            if sum(valid_cameras) < 2 % Less than 2 cameras used
                camsused = 'None'; % Not enough cameras for reconstruction
            else
                camsused = strjoin(camera_labels(valid_cameras), ''); % Concatenate valid camera labels
            end

            % Store results for this time point
            H(k, :) = [g(1), g(2), g(3), avgres]; % First four numeric columns
            camsused_cell{k} = camsused;          % Separate cell array for cameras used
        end
        % Combine H and camsused_cell into a cell array for this point
        H_all{p} = [num2cell(H), camsused_cell];

        % Remove the first two rows to eliminate invalid data
        H_all{p}(1:2, :) = [];

    end
end




% Perform reconstruction
H_all = RECONFU_parallel(A, L_matrices);

% Display results for Point 1
disp('Results for Point 1:');
disp(H_all{1});



% Define the output folder and file name dynamically
output_folder = '/Users/viggorey/Desktop/PhD/Cambridge/Macaranga/3D transformation/5. Datasets/3D data/Large branch/';
output_file = fullfile(output_folder, [file_prefix, '.xlsx']); % Dynamic file name

% Loop through each tracking point
for point = 1:num_points
    % Prepare data for export
    time_frames = (1:size(H_all{point}, 1))'; % Time frame indices
    X = cell2mat(H_all{point}(:, 1)); % Extract X
    Y = cell2mat(H_all{point}(:, 2)); % Extract Y
    Z = cell2mat(H_all{point}(:, 3)); % Extract Z
    Residual = cell2mat(H_all{point}(:, 4)); % Extract Residual
    CamerasUsed = H_all{point}(:, 5); % Extract Cameras Used (as strings)

    % Combine data into a table
    export_data = table(time_frames, X, Y, Z, Residual, CamerasUsed, ...
        'VariableNames', {'Time Frame', 'X', 'Y', 'Z', 'Residual', 'Cameras Used'});

    % Write to a new sheet in the Excel file
    writetable(export_data, output_file, 'Sheet', ['Point ', num2str(point)]);
end

disp(['Data successfully exported to ', output_file]);


figure;
hold on;

% Create a colormap with distinct colors for all points
cmap = lines(num_points); % Use MATLAB's built-in 'lines' colormap

% Loop through all points in H_all
for point = 1:num_points
    % Extract the numerical 3D coordinates (X, Y, Z) from the numeric columns
    X = cell2mat(H_all{point}(:, 1)); % Column 1: X
    Y = cell2mat(H_all{point}(:, 2)); % Column 2: Y
    Z = cell2mat(H_all{point}(:, 3)); % Column 3: Z

    % Plot the trajectory for the current point with a unique color
    plot3(X, Y, Z, 'o-', 'Color', cmap(point, :), 'DisplayName', ['Point ', num2str(point)]);
end

% Add labels, title, legend, and grid
xlabel('X');
ylabel('Y');
zlabel('Z');
title('3D Trajectories of All Tracking Points');
legend; % Show legend with all points
grid on;

hold off;