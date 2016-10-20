% Class for applying the Lucas-Kanade algorithm that we are studying in
% as an object class. Goal here is for students to translate this
% code into a C++ class, using OpenCV, that can run in Xcode. 
%
% Inherits properties from LK class. 
%
%  Created by Simon Lucey 
%--------------------------------------------------------------------------
%
% Usage:- obj = LK_IC(tmplt)
%
% where:- 
%      <in> tmplt = template to be used within the LK framework
%     <out> obj = returned LK_IC object
%
classdef LK_IC < LK           
    properties
            % Define other properties
            R; % Update matrix that is to be pre-computed
    end
    %---------------------------------------------------------------------
    % Method for executing the LK algorithm 
    methods
        %% ----------------------------------------------------------------
        % Constructor function for the LK class, initializes everything
        % with the template we want to match against for LK. 
        function o = LK_IC(tmplt)        
            
            % Execute the mother constructor
            o@LK(tmplt); 

            % Pre-compute the Gaussian blur filter
            o.T_0 = imfilter(o.T_0, o.fb,'replicate'); % Blur the template before hand
            
            Tx_0 = imfilter(o.T_0, o.fx,'replicate'); % Get the x-gradient
            Ty_0 = imfilter(o.T_0, o.fy,'replicate'); % Get the y-gradient

            % Form the Jacobian with the current estimate of p
            [dWx,dWy] = dWx_dp(o.w, o.x, o.y); % Estimate the derivative of the warp w.r.t p
                
            % Apply the gradients with the derivative of the warp (i.e chain rule style as covered in class)
            J = dWx.*repmat(Tx_0(:),[1,o.w.P]) + dWy.*repmat(Ty_0(:),[1,o.w.P]);
            
            % Pre-compute the update matrix
            o.R = inv(J'*J)*J'; 
            
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
            
            
            % Now apply number of iterations
            for n = 1:num_iter               
                % Visualize the result if h is not empty
                if ~isempty(h)
                    redraw(o.w, p, o.dsize, h); 
                    fprintf('Waiting for a key press....\n',n); 
                    pause; % Wait for the key press
                end
                
                I_p = imwarp(o.w, img, p, o.dsize); % Warp the image
                
                % Get the error and display the result
                diff = I_p(:) - o.T_0(:); 
                fprintf('Iter: %d \t Err: %g\n',n, sqrt(diff'*diff));  
                
                % Apply the update
                dp = o.R*(I_p(:) - o.T_0(:)); 
                
                % Step 5. inverse compositional step
                %
                % You have to work this out yourself using 
                % M = p2M(o.w, p); and also p = M2p(o.w, M); 
                
            end
        end
    end
end

