% Example code for Assignment 3 in 16423
rgb = imread('affine_prince.jpg'); 
img = rgb2gray(rgb); 

% Choose to scale the image down
sc = 0.3; img = imresize(img,sc); 

% 2D projected points on the book
pts = sc*[771, 1587, 771, 1587;
       885, 885, 1503, 1503]; 
 
% Height and width of the template
dsize = [30,30]; 
   
% Set the template points (in order that points appear in image)
tmplt_pts = [0, dsize(2)-1, 0, dsize(2)-1; 
             0, 0, dsize(1)-1, dsize(1)-1]; 

% Step 1. Extract the Template
a = Affine; % Define an Affine object
gnd_p = a.fit(tmplt_pts, pts); % Get the ground-truth warp
T_0 = a.imwarp(img, gnd_p, dsize); % Define the template
clf; imagesc(img); axis off; axis image; % Display the image
a.draw(gnd_p, dsize,'r-','LineWidth',2); % Draw the ground-truth

% Step 2. Perturb the warp 
dp = [0.26, 0.26, -18.15, -0.62, -0.36, -2.32]'; 
noise_p = gnd_p + dp; 
h = a.draw(noise_p, dsize,'b-','LineWidth',2); % Draw the ground-truth

% Step 3. Apply the LK algorithm
lk = LK_IC(T_0); 
lk.fit(img, noise_p, 30, h); % Append h here to draw the result on the image

% Step 4. Get the timing info
tic; 
lk.fit(img, noise_p, 30);
toc; 


         