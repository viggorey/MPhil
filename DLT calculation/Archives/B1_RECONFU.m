% Define the DLT coefficients (A) and the camera coordinates (L)
A = [
    88.2106, 93.4303, 78.7655, 96.0516; % Camera 1 - first coefficient for all cameras
    -6.6498, -14.1814, -3.1569, -10.0825; % Camera 2
    37.76, -2.7843, -45.5904, -8.2196; 
    608.6661, 378.2291, 374.2239, 348.6168; 
    -2.8612, 1.9343, -0.1604, -1.9495; 
    140.3325, 131.9784, 134.1519, 91.1523; 
    0.6305, 1.0066, 7.2696, -71.4079; 
    528.5439, 500.3596, 502.5081, 342.7244; 
    0.0011, 0.001, 0.0015, 0.0031; 
    -0.013, -0.0208, -0.0149, -0.0185; 
    -0.0017, -0.0006, 0.0019, -0.0076
];

L = [
    625.1381, 743.1407, 595.6553, 508.7630, 411.3907, 541.8794, 372.3026, 472.8909;  % Time 1
    779.5875, 725.3298, 616.2684, 520.4482, 349.5512, 567.1768, 160.6014, 529.9377;  % Time 2
    765.9793, 542.6374, 642.3663, 335.9553, 358.8968, 376.1215, 162.8252, 547.7748;  % Time 3
    595.7903, 557.6862, 619.6047, 323.7356, 422.1993, 351.7099, 375.7144, 490.4268   % Time 4
];


% Step 3: Define the Cut points (if any). In this case, no points to cut.
Cut = [];  % Optional points to exclude (if needed)

% Step 4: Call the RECONFU function to calculate the 3D coordinates
% (You need to define the RECONFU function as shown in the previous response)
[H] = RECONFUcalc(A, L);

% Step 5: Display the reconstructed 3D coordinates, residuals, and cameras used
disp('Reconstructed 3D Coordinates and Residuals:');
disp('3D Coordinates (X, Y, Z), Residuals, Cameras Used:');
disp(H);

% Function to calculate the 3D coordinates from DLT coefficients
function [H] = RECONFUcalc(A, L)
    % Description: Reconstruction of 3D coordinates with the use of local
    % camera coordinates and the DLT coefficients for n cameras.
    
    n = size(A,2);
    % Check whether the numbers of cameras agree for A and L
    if 2*n ~= size(L,2)
        disp('The number of cameras in A and L do not match.')
        disp('Press any key to try again');
        pause; return
    end

    H = zeros(size(L,1), 5); % Initialize H matrix to store 3D coordinates, residuals, and cameras used

    % Iterate over each time point
    for k = 1:size(L,1)
        q = 0; L1 = []; L2 = []; % Initialize matrices for calculations
        for i = 1:n % Iterate through the number of cameras
            x = L(k, 2*i-1); y = L(k, 2*i); % Get camera x and y coordinates
            if ~(isnan(x) || isnan(y))  % Only use valid points
                q = q + 1; % Count valid cameras
                % Construct the L1 and L2 matrices
                L1([q*2-1:q*2], :) = [A(1,i) - x*A(9,i), A(2,i) - x*A(10,i), A(3,i) - x*A(11,i); ...
                                      A(5,i) - y*A(9,i), A(6,i) - y*A(10,i), A(7,i) - y*A(11,i)];
                L2([q*2-1:q*2], :) = [x - A(4,i); y - A(8,i)];
            end
        end

        % Check if there are enough valid cameras (at least 2)
        if (size(L2,1)/2) > 1
            g = L1\L2; % Solve for the 3D coordinates
            h = L1*g; % Recalculate the 2D coordinates
            DOF = (size(L2,1) - 3); % Degrees of freedom
            avgres = sqrt(sum([L2 - h].^2) / DOF); % Compute average residuals
        else
            g = [NaN; NaN; NaN]; avgres = NaN;
        end
        
        % Identify which cameras were used for the reconstruction
        b = fliplr(find(sum(reshape(isnan(L(k,:)), 2, size(L(k,:),2)/2)) == 0));
        if size(b,2) < 2
            camsused = NaN;
        else
            for w = 1:size(b,2)
                b(1,w) = b(1,w) * 10^(w-1);
            end
            camsused = sum(b');
        end

        % Store the results: 3D coordinates, residuals, and cameras used
        H(k, :) = [g', avgres, camsused];
    end
end