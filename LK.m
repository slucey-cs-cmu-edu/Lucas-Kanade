% Class for applying the Lucas-Kanade algorithm that we are studying in
% 16423 as an object class. Goal here is for students to translate this
% code into a C++ class, using OpenCV, that can run in Xcode. 
%
% Should be easy to use, simply perform
%
% >> imagesc(img); 
% >> LK lk(T_0); 
% >> p = lk.fit(img, p0, 10); % Fit with LK
%
% Written by Simon Lucey 2015
classdef LK            
    % Define other variables need for calibration
    properties
            % Define other properties
            fx; % Kernel for Sobel filter in x- direction
            fy; % Kernel for Sobel filter in y- direction
            fb; % Kernel for Gaussian blurring filter
            x; % x-coordinates
            y; % y-coordinates
            T_0; % Template appearance we are going to match against (at warp p = 0)
            dsize; % Size of the template
            w; % Set the warp object (can be Affine or Homography) 
    end
    %---------------------------------------------------------------------
    % Method for executing the LK algorithm 
    methods
        %% ----------------------------------------------------------------
        % Constructor function for the LK class, initializes everything
        % with the template we want to match against for LK. 
        function o = LK(tmplt)        
            % Pre-compute Sobel filters in x- and y- directions
            o.fx = [-1,1]; o.fy = o.fx'; 
            
            % Pre-compute the Gaussian blur filter
            o.fb = fspecial('gaussian',[5,5], 3);   
            
            % Set the template
            if isinteger(tmplt)
                tmplt = im2double(tmplt); 
            end
            o.T_0 = tmplt; 
            o.dsize = size(tmplt); 
            
            % Set the pixel coordinates within the template 
            [x,y] = meshgrid(0:size(tmplt,2)-1,0:size(tmplt,1)-1);
            o.x = x(:); o.y = y(:); % Store the vectorized coordinates
            
            % Set it to be an affine warp (internally for now)
            o.w = Affine; 
        end        
        %% ----------------------------------------------------------------
        % Fit the tmplt to the image using the initial warp in pinit
        function p = fit(o, img, pinit, num_iter, varargin)
  
            % Initialize p with pinit
            p = pinit; 
  
            % Ensure image is a double
            if isinteger(img)
                img = im2double(img); 
            end
            
            % Get handle to the plot
            h = []; 
            if nargin > 4
                h = varargin{1}; 
            end
            
            % Prec-computer the gradients on the image
            bimg = imfilter(img, o.fb); % Blur image before hand
            Ix = imfilter(bimg, o.fx); % Get the x-gradient
            Iy = imfilter(bimg, o.fy); % Get the y-gradient
            
            % Now apply number of iterations
            for n = 1:num_iter               
                % Visualize the result if h is not empty
                if ~isempty(h)
                    redraw(o.w, p, o.dsize, h); 
                    fprintf('Waiting for a key press....\n',n); 
                    pause; % Wait for the key press
                end
                
                % Step 1. warp the images and the gradients 
                I_p = imwarp(o.w, bimg, p, o.dsize); % Warp the image
                Ix_p = imwarp(o.w, Ix, p, o.dsize); % Warp the x-gradient
                Iy_p = imwarp(o.w, Iy, p, o.dsize); % Warp the y-gradient
                
                % Get the error and display the result
                diff = I_p(:) - o.T_0(:); 
                fprintf('Iter: %d \t Err: %g\n',n, sqrt(diff'*diff));  
                
                % Step 2. Form the Jacobian with the current estimate of p
                [dWx,dWy] = dWx_dp(o.w, o.x, o.y, p); % Estimate the derivative of the warp w.r.t p
                
                % Step 3.  the image gradients with the derivative (i.e chain
                % rule style as covered in class)
                J = dWx.*repmat(Ix_p(:),[1,o.w.P]) + dWy.*repmat(Iy_p(:),[1,o.w.P]);
                
                % Step 4. now solve the linear system
                dp = J\(o.T_0(:) - I_p(:)); 
                
                % Step 5. update
                p = p + dp;                 
            end
        end
    end
end

