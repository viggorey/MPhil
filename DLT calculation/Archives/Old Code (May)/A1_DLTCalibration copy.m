% Define your 3D coordinates (F)
F = [
    0, 0, 0;
    0, 0, -2;
    0, 2, -2;
    0, 2, 0;
    2, 0, -2;
    2, 2, -2;
    2, 0, 0;
    2, 2, 0
];  % Example 3D calibration points

% Define your 2D coordinates (L)
L = [
    757.61, 317.74;
    740.83, 315.51;
    734.32, 608.77;
    750.84, 610.45;
    923.88, 319.96;
    917.02, 611.40;
    941.06, 320.06;
    934.44, 613.62
]; % Corresponding 2D coordinates in the image

% Optional: Define Cut points if any (this is where you exclude points)
Cut = [];  % Or you can specify like Cut = [1, 3];

% Call the function to get the DLT coefficients and residuals
[A, avgres] = calculateDLT(F, L, Cut);

% Display the results
disp('DLT Coefficients:');
disp(A);
disp('Average Residuals:');
disp(avgres);

% Function to calculate the DLT coefficients
function [A, avgres] = calculateDLT(F, L, Cut)
    % Description:	Program to calculate DLT coefficient for one camera
    % Note that at least 6 (valid) calibration points are needed
    % function [A,avgres] = dltfu(F,L,Cut)
    % Input:	- F      matrix containing the global coordinates (X,Y,Z)
    %                        of the calibration frame
    %			 e.g.: [0 0 20;0 0 50;0 0 100;0 60 20 ...]
    %		- L      matrix containing 2d coordinates of calibration 
    %                        points seen in camera (same sequence as in F)
    %                        e.g.: [1200 1040; 1200 1360; ...]
    %               - Cut    points that are not visible in camera;
    %                        not being used to calculate DLT coefficient
    %                        e.g.: [1 7] -> calibration point 1 and 7 
    %			 will be discarded.
    %		      	 This input is optional (default Cut=[]) 
    % Output:	- A      11 DLT coefficients
    %               - avgres average residuals (measure for fit of dlt)
    %			 given in units of camera coordinates
    %
    % Author:	Christoph Reinschmidt, HPL, The University of Calgary
    % Date:		January, 1994
    % Last changes: November 29, 1996
    % Version:	1.0
    % References:	Woltring and Huiskes (1990) Stereophotogrammetry. In
    %               Biomechanics of Human Movement (Edited by Berme and
    %               Cappozzo). pp. 108-127.

    if nargin == 2; Cut = []; end;

    if size(F,1) ~= size(L,1)
        disp('# of calibration points entered and seen in camera do not agree'), return
    end

    m = size(F,1); Lt = L'; C = Lt(:); % Flatten the 2D points into a column vector

    % Build the B matrix
    for i = 1:m
        B(2*i-1,1)  = F(i,1); % x
        B(2*i-1,2)  = F(i,2); % y
        B(2*i-1,3)  = F(i,3); % z
        B(2*i-1,4)  = 1;
        B(2*i-1,9)  = -F(i,1)*L(i,1); % -x * u
        B(2*i-1,10) = -F(i,2)*L(i,1); % -y * u
        B(2*i-1,11) = -F(i,3)*L(i,1); % -z * u
        B(2*i,5)    = F(i,1); % x
        B(2*i,6)    = F(i,2); % y
        B(2*i,7)    = F(i,3); % z
        B(2*i,8)    = 1;
        B(2*i,9)    = -F(i,1)*L(i,2); % -x * v
        B(2*i,10)   = -F(i,2)*L(i,2); % -y * v
        B(2*i,11)   = -F(i,3)*L(i,2); % -z * v
    end

    % Remove the points that need to be discarded
    if ~isempty(Cut)
        Cutlines = [Cut.*2-1, Cut.*2];
        B(Cutlines,:) = [];
        C(Cutlines,:) = [];
    end

    % Solve for the DLT coefficients
    A = B\C; % Solve the matrix equation
    D = B * A; % Back-calculate the u,v image coordinates
    R = C - D; % Calculate residuals
    res = norm(R); 
    avgres = res / sqrt(size(R,1)); % Calculate average residuals
end

%what is c and R and what norm does