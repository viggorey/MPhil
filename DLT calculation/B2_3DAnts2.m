% Define: 
base_path = '/Users/viggorey/Desktop/PhD/Cambridge/Macaranga/3D transformation/5. Datasets/2D data/'; % Base path
file_prefix = '11U1'; % Common file prefix
camera_suffixes = {'L', 'T', 'R', 'F'}; % Camera suffixes (Top, Left, Right, Front)


A = [
90.798, 91.6283, 68.6801, -6.9384; %  1 - first coefficient for all cameras
0.9664, -2.8374, 5.6441, 114.8824;
24.4139, -17.6627, -59.2024, -29.8135;
594.3277, 667.1884, 676.3614, 498.9457;
-1.7397, 1.7846, -0.6094, -77.332;
149.6519, 147.3786, 143.9186, -3.9629;
0.121, -0.7585, 4.8412, 8.6032;
331.1221, 300.1799, 342.79, 590.4905;
0.0006, 0.001, -0.0011, -0.001;
-0.0012, 0.0003, 0.0006, 0.0003;
-0.0005, -0.0019, -0.0013, -0.0011
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

function [H_all, combo_stats] = RECONFU_parallel(A, L_matrices)
    % Get the number of points to process
    num_points = length(L_matrices);
    
    % Preallocate outputs
    H_all = cell(1, num_points);
    combo_stats = struct();
    
    % Start parallel processing
    parfor p = 1:num_points
        % Initialize statistics collection for this point
        point_stats = struct();
        
        % Extract the L matrix for the current point
        L = L_matrices{p};
        n = size(A, 2);
        
        if 2 * n ~= size(L, 2)
            disp('The # of cameras given in A and L do not agree');
            H_all{p} = NaN;
            continue;
        end
        
        % Initialize outputs
        H = NaN(size(L, 1), 4);
        camsused_cell = cell(size(L, 1), 1);
        % Corrected camera labels order to match LTRF
        camera_labels = {'L', 'T', 'R', 'F'};
        
        % Process each time frame
        for frame_idx = 1:size(L, 1)
            valid_cameras = [];
            for i = 1:n
                if ~(isnan(L(frame_idx, 2*i-1)) || isnan(L(frame_idx, 2*i)))
                    valid_cameras = [valid_cameras, i];
                end
            end
            
            if length(valid_cameras) >= 2
                % Track results for all combinations
                all_results = [];  % Will be a struct array
                result_idx = 1;
                
                % Generate all possible camera combinations
                camera_numbers = 2:length(valid_cameras);
                for cam_num = camera_numbers
                    cam_combos = nchoosek(valid_cameras, cam_num);
                    
                    for combo_idx = 1:size(cam_combos, 1)
                        current_combo = cam_combos(combo_idx, :);
                        
                        % Build matrices
                        q = 0;
                        L1 = [];
                        L2 = [];
                        
                        for cam_idx = current_combo
                            x = L(frame_idx, 2*cam_idx-1);
                            y = L(frame_idx, 2*cam_idx);
                            
                            q = q + 1;
                            L1([q*2-1:q*2], :) = [A(1,cam_idx) - x*A(9,cam_idx), A(2,cam_idx) - x*A(10,cam_idx), A(3,cam_idx) - x*A(11,cam_idx);
                                                 A(5,cam_idx) - y*A(9,cam_idx), A(6,cam_idx) - y*A(10,cam_idx), A(7,cam_idx) - y*A(11,cam_idx)];
                            L2([q*2-1:q*2], :) = [x - A(4,cam_idx); y - A(8,cam_idx)];
                        end
                        
                        % Reconstruction
                        g = L1 \ L2;
                        h = L1 * g;
                        
                        % Calculate DOF-based residual
                        DOF = (length(current_combo) * 2 - 3);  % Degrees of freedom
                        residual = sqrt(sum([L2 - h].^2) / DOF);
                        
                        % Calculate geometric score
                        geometric_score = calculateGeometricScore(current_combo);
                        
                        % Store result
                        cam_labels = camera_labels(current_combo);
                        combo_name = strjoin(cam_labels, '');
                        
                        % Add to results array
                        all_results(result_idx).combo = combo_name;
                        all_results(result_idx).g = g;
                        all_results(result_idx).residual = residual;
                        all_results(result_idx).num_cams = length(current_combo);
                        all_results(result_idx).geometric_score = geometric_score;
                        result_idx = result_idx + 1;
                        
                        % Collect statistics
                        if ~isfield(point_stats, combo_name)
                            point_stats.(combo_name) = struct('residuals', [], 'num_cams', length(current_combo));
                        end
                        point_stats.(combo_name).residuals = [point_stats.(combo_name).residuals, residual];
                    end
                end
                
                if ~isempty(all_results)
                    % Extract arrays for scoring
                    residuals = [all_results.residual];
                    num_cams = [all_results.num_cams];
                    geometric_scores = [all_results.geometric_score];
                    
                    % Normalize scores
                    norm_residuals = residuals / max(residuals);
                    norm_cams = num_cams / max(num_cams);
                    norm_geometric = geometric_scores / max(geometric_scores);
                    
                    % Combined score (weighted sum)
                    combined_scores = -0.7 * norm_residuals + 0.15 * norm_cams + 0.15 * norm_geometric;
                    
                    [~, best_idx] = max(combined_scores);
                    best_result = all_results(best_idx);
                    
                    % Store results
                    H(frame_idx, :) = [best_result.g(1), best_result.g(2), best_result.g(3), best_result.residual];
                    camsused_cell{frame_idx} = best_result.combo;
                else
                    H(frame_idx, :) = [NaN, NaN, NaN, NaN];
                    camsused_cell{frame_idx} = 'None';
                end
            else
                H(frame_idx, :) = [NaN, NaN, NaN, NaN];
                camsused_cell{frame_idx} = 'None';
            end
        end
        
        % Store results for this point
        H_all{p} = [num2cell(H), camsused_cell];
        H_all{p}(1:2, :) = []; % Remove first two rows
        
        % Store statistics for this point
        combo_stats(p).point = p;
        combo_stats(p).stats = point_stats;
    end
end

function score = calculateGeometricScore(camera_indices)
    % Define approximate camera positions (corrected order for LTRF)
    camera_positions = {
        [-0.5, 0.866, 0],  % Left (30 degrees to left)
        [0, 1, 0],         % Top (directly above)
        [0.5, 0.866, 0],   % Right (30 degrees to right)
        [0, 0.940, 0.342]  % Front (20 degrees above)
    };
    
    % Calculate angles between cameras
    angles = [];
    for i = 1:length(camera_indices)
        for j = (i+1):length(camera_indices)
            v1 = camera_positions{camera_indices(i)};
            v2 = camera_positions{camera_indices(j)};
            % Calculate angle between camera vectors
            angle = acosd(dot(v1, v2) / (norm(v1) * norm(v2)));
            angles = [angles, angle];
        end
    end
    
    % Score based on angular distribution (closer to 90 degrees is better)
    score = mean(1 - abs(angles - 90)/90);
end

function plotResidualComparisons(combo_stats, H_all)
    % Create figure
    figure('Position', [100, 100, 1200, 800]);
    
    % Get all unique combinations
    all_combos = {};
    for p = 1:length(combo_stats)
        combos = fieldnames(combo_stats(p).stats);
        all_combos = union(all_combos, combos);
    end
    
    % 1. Distribution plot of residuals by camera combination
    subplot(2,2,1)
    residuals_by_combo = cell(length(all_combos), 1);
    for i = 1:length(all_combos)
        combo = all_combos{i};
        all_residuals = [];
        for p = 1:length(combo_stats)
            if isfield(combo_stats(p).stats, combo)
                all_residuals = [all_residuals, combo_stats(p).stats.(combo).residuals];
            end
        end
        residuals_by_combo{i} = all_residuals;
    end
    
    % Create violin plot-like visualization
    hold on;
    for i = 1:length(residuals_by_combo)
        data = residuals_by_combo{i};
        if ~isempty(data)
            % Calculate statistics
            med = median(data);
            q1 = quantile(data, 0.25);
            q3 = quantile(data, 0.75);
            
            % Plot median point
            plot(i, med, 'k.', 'MarkerSize', 15);
            
            % Plot IQR box
            rectangle('Position', [i-0.25, q1, 0.5, q3-q1], 'FaceColor', [0.8 0.8 0.8]);
            
            % Plot vertical line for range
            line([i i], [min(data) max(data)], 'Color', 'k');
        end
    end
    hold off;
    
    set(gca, 'XTick', 1:length(all_combos));
    set(gca, 'XTickLabel', all_combos);
    xtickangle(45);
    title('Residuals Distribution by Camera Combination');
    ylabel('Normalized Residual');
    xlabel('Camera Combination');
    
    % 2. Bar plot of average residuals by number of cameras
    subplot(2,2,2)
    num_cams_stats = struct();
    for i = 1:length(all_combos)
        combo = all_combos{i};
        num_cams = combo_stats(1).stats.(combo).num_cams;
        if ~isfield(num_cams_stats, ['n', num2str(num_cams)])
            num_cams_stats.(['n', num2str(num_cams)]) = [];
        end
        for p = 1:length(combo_stats)
            if isfield(combo_stats(p).stats, combo)
                num_cams_stats.(['n', num2str(num_cams)]) = [num_cams_stats.(['n', num2str(num_cams)]), ...
                    mean(combo_stats(p).stats.(combo).residuals)];
            end
        end
    end
    
    num_cams_means = [];
    num_cams_stds = [];
    cam_numbers = [];
    for i = 2:4  % Assuming 2-4 cameras
        field = ['n', num2str(i)];
        if isfield(num_cams_stats, field)
            num_cams_means = [num_cams_means, mean(num_cams_stats.(field))];
            num_cams_stds = [num_cams_stds, std(num_cams_stats.(field))];
            cam_numbers = [cam_numbers, i];
        end
    end
    
    bar(cam_numbers, num_cams_means);
    hold on;
    errorbar(cam_numbers, num_cams_means, num_cams_stds, 'k.');
    title('Average Residual by Number of Cameras');
    xlabel('Number of Cameras');
    ylabel('Average Normalized Residual');
    
    % 3. Usage frequency of different combinations
    subplot(2,2,3)
    combo_counts = zeros(length(all_combos), 1);
    for p = 1:length(combo_stats)
        results = H_all{p};
        combos_used = results(:,5);
        for i = 1:length(all_combos)
            combo_counts(i) = combo_counts(i) + sum(strcmp(combos_used, all_combos{i}));
        end
    end
    
    bar(combo_counts);
    set(gca, 'XTick', 1:length(all_combos));
    set(gca, 'XTickLabel', all_combos);
    xtickangle(45);
    title('Usage Frequency of Camera Combinations');
    xlabel('Camera Combination');
    ylabel('Frequency');
    
    % 4. NEW: Time series of residuals for each point
    subplot(2,2,4)
    hold on;
    
    % Create a colormap for different points
    num_points = length(H_all);
    colors = lines(num_points);
    
    % Plot residuals over time for each point
    for p = 1:num_points
        % Extract residuals from H_all
        residuals = cell2mat(H_all{p}(:, 4));
        time_points = 1:length(residuals);
        
        % Plot with unique color and label
        plot(time_points, residuals, 'Color', colors(p,:), 'DisplayName', ['Point ' num2str(p)]);
    end
    
    title('Residuals Over Time for Each Point');
    xlabel('Frame Number');
    ylabel('Residual');
    legend('show');
    grid on;
    
    % Add overall title
    sgtitle('Camera Combination Analysis');
end



% Perform reconstruction with statistics collection
[H_all, combo_stats] = RECONFU_parallel(A, L_matrices);

% Display results for Point 1
disp('Results for Point 1:');
disp(H_all{1});

% Plot the residual comparisons
plotResidualComparisons(combo_stats, H_all);



% Define the output folder and file name dynamically
output_folder = '/Users/viggorey/Desktop/PhD/Cambridge/Macaranga/3D transformation/5. Datasets/3D data/';
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


