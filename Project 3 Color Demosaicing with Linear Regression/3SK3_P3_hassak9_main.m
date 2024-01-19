clc; clear;
%% Image Inputs & Paramaters %% 
training_image = double(imread("training.jpg"));

% Uncomment out 1 @ a time (1/4)
%%% Non-Bayer Input
% testing_image = im2double(imread("testing_image_1.jpg"));
% testing_image = im2double(imread("testing_image_2.jpg"));
% testing_image = im2double(imread("testing_image_3.jpg"));

%%% Bayer Image Input:
% testing_image = im2double(imread("test1.png"));
% testing_image = im2double(imread("test2.png"));

% note: im2double is better when the image will be displayed. TODO undo
% normalization to [0, 1]

patch_size = [1 1];
window_size = [5 5];
pad_size = [2 2];


%% Step 1: Generate Coefficient Matrices %%
% Extract red, green and blue channels, pad to acommodate for processing
% image edges
red = training_image(:,:,1);
red_col = im2col(red, patch_size);
red_pad = padarray(red, pad_size, 'symmetric', 'both');
red_win = im2col(red_pad, window_size);

green = training_image(:,:,2);
green_col = im2col(green, patch_size);
green_pad = padarray(green, pad_size, 'symmetric', 'both');
green_win = im2col(green_pad, window_size); 

blue= training_image(:,:,3);
blue_col = im2col(blue, patch_size); 
blue_pad = padarray(blue, pad_size, 'symmetric', 'both');
blue_win = im2col(blue_pad, window_size); 

% acquire coefficient matrices, 2 for each possible type of mosaic patch
[green_A_RGGB, blue_A_RGGB] = RGGB(red_win, green_win, blue_win, green_col,blue_col);
[green_A_BGGR,red_A_BGGR] = BGGR(red_win, green_win, blue_win, green_col, red_col);
[red_A_GRBG, blue_A_GRBG] = GRBG(red_win, green_win, blue_win, red_col,blue_col);
[blue_A_GBRG, red_A_GBRG] = GBRG(red_win, green_win, blue_win, blue_col,red_col);

%% Step 2: Testing
[x, y, z] = size(testing_image);

% Uncomment out 1 @ a time (2/4)
%%% Non-Bayer Input
% For RGGB Images:
% bayer_image = testing_image(:,:,2);
% bayer_image(1:2:x, 1:2:y) = testing_image(1:2:x, 1:2:y, 1);
% bayer_image(2:2:x, 2:2:y) = testing_image(2:2:x, 2:2:y, 3);
% greyscale = bayer_image;
% imwrite(bayer_image, "results/bayer_image.jpg")

%%% Bayer Image Input:
bayer_image = testing_image;
greyscale = testing_image;

if (rem(x, 2) ~= 0)% need to pad 1 row of size 510
    bayer_image(end+1, :) = zeros(1, 510);
end


%% Benchmarking
% Uncomment out 1 @ a time (3/4)
%%% Non-Bayer Input
% builtin_demosaic = demosaic(imread("results/bayer_image.jpg"), "rggb");
% RMSE_BI = get_RMSE(im2uint8(builtin_demosaic), im2uint8(testing_image));

%%% Bayer Image Input:
builtin_demosaic = demosaic(imread("test1.png"), "rggb");
% builtin_demosaic = demosaic(imread("test2.png"), "rggb");

imwrite(builtin_demosaic, "results/builtin_demosaic.jpg");

%% Step 3: Linear Regression
% reconstruct channels from Bayer Image
recon_r = repmat([1 0; 0 0], ceil(x/2), ceil(y/2)) .* bayer_image;
recon_g = repmat([0 1; 1 0], ceil(x/2), ceil(y/2)) .* bayer_image;
recon_b = repmat([0 0; 0 1], ceil(x/2), ceil(y/2)) .* bayer_image;

bayer_image = padarray(bayer_image, [3 3], 'symmetric', 'both');
bayer_image(end - 2, :) = []; 
bayer_image(:, end - 2) = [];
bayer_image(3, :) = []; 
bayer_image(:, 3) = [];

for i = 3:x + 2
    for j = 3:y + 2
        % extract and flatten a 5x5 submatrix of greyscale bayer_image
        flatten = bayer_image(i - 2:i + 2, j - 2:j + 2);
        flatten = flatten(:);
        
        if mod(i, 2) && ~mod(j, 2) % red, use GRBG to reconstruct
            recon_r(i - 2, j - 2) = red_A_GRBG'*flatten;
            recon_b(i - 2, j - 2) = blue_A_GRBG'*flatten;
        
        elseif ~mod(i, 2) && ~mod(j, 2) % green, use BGGR to reconstruct
            recon_g(i - 2, j - 2) = green_A_BGGR'*flatten;
            recon_r(i - 2, j - 2) = red_A_BGGR'*flatten;
        
        elseif mod(i, 2) && mod(j, 2) % blue, use RGGB to reconstruct
            recon_b(i - 2, j - 2) = blue_A_RGGB'*flatten;
            recon_g(i - 2, j - 2) = green_A_RGGB'*flatten;
        
        elseif ~mod(i, 2) && mod(j, 2) % blue, use GBRG to reconstruct
            recon_b(i - 2, j - 2) = blue_A_GBRG'*flatten;
            recon_r(i - 2, j - 2) = red_A_GBRG'*flatten;
        end
    end
end

reconstructed_image = cat(3, recon_r, recon_g, recon_b);
reconstructed_image = uint8(reconstructed_image.*255); % undoing normalization
imwrite(reconstructed_image, "results/linear_regression_reconstruction.jpg");

% Uncomment out 1 @ a time (4/4)
%%% Display RMSE Results %%%
%%% Non-Bayer Input
% RMSE_LR = get_RMSE(im2uint8(reconstructed_image), im2uint8(testing_image));
% fprintf("RMSE b/w testing image and reconstructed: %.10f\n", RMSE_LR);
% fprintf("RMSE b/w testinag image and builtin: %.10f\n", RMSE_BI);

% subplot to view results
figure;
subplot(1,3,1);
imshow(greyscale);
title('Bayer Image');
subplot(1,3,2);
imshow(reconstructed_image);
title('Reconstructed using Linear Regression');
subplot(1,3,3);
imshow(builtin_demosaic);
title('Built In');


%% Utility Functions %%

function RMSE = get_RMSE(image_1, image_2)
    RMSE = sqrt(immse(image_1, image_2));
end

function [green_A_RGGB, blue_A_RGGB] = RGGB(red_win,green_win,blue_win,green_col,blue_col)

    red_pat = repmat ([1 0;0 0], [3,3]); 
    green_pat = repmat ([0 1;1 0], [3,3]);
    blue_pat = repmat ([0 0;0 1], [3,3]); 

    red_pat(6,:) = []; 
    red_pat(:,6) = [];

    green_pat(6,:) = []; 
    green_pat(:,6) = []; 

    blue_pat(6,:) = [];
    blue_pat(:,6) = [];

    X = (im2col(red_pat,[1 1])'.*red_win+im2col(green_pat,[1 1])'.*green_win+im2col(blue_pat,[1 1])'.*blue_win)';

    blue_A_RGGB = inv(X'*X)*X'*(blue_col');
    green_A_RGGB = inv(X'*X)*X'*(green_col');
end

function [green_A_BGGR,red_A_BGGR] = BGGR(red_win,green_win,blue_win,green_col,red_col)

    blue_pat = repmat ([1 0;0 0], [3,3]);
    green_pat = repmat ([0 1;1 0], [3,3]);
    red_pat = repmat ([0 0;0 1], [3,3]); 

    red_pat(6,:) = []; 
    red_pat(:,6) = []; 
    
    green_pat(6,:) = []; 
    green_pat(:,6) = []; 
    
    blue_pat(6,:) = [];
    blue_pat(:,6) = [];

    X = (im2col(red_pat,[1,1])'.*red_win+im2col(green_pat,[1,1])'.*green_win+im2col(blue_pat,[1,1])'.*blue_win)';

    red_A_BGGR = inv(X'*X)*X'*(red_col');
    green_A_BGGR = inv(X'*X)*X'*(green_col');
end

function [red_A_GRBG,blue_A_GRBG] = GRBG(red_win,green_win,blue_win,red_col,blue_col)

    green_pat = repmat ([1 0;0 1], [3,3]);
    blue_pat= repmat ([0 0;1 0], [3,3]);
    red_pat= repmat ([0 1;0 0], [3,3]); 

    red_pat(6,:) = []; 
    red_pat(:,6) = []; 
    
    green_pat(6,:) = []; 
    green_pat(:,6) = []; 
    
    blue_pat(6,:) = [];
    blue_pat(:,6) = [];

    X = (im2col(red_pat,[1,1])'.*red_win+im2col(green_pat,[1,1])'.*green_win+im2col(blue_pat,[1,1])'.*blue_win)';

    blue_A_GRBG = inv(X'*X)*X'*(blue_col');
    red_A_GRBG = inv(X'*X)*X'*(red_col');
end

function [blue_A_GBRG,red_A_GBRG] = GBRG(red_win,green_win,blue_win,blue_col,red_col)
    
    green_pat= repmat ([1 0;0 1], [3,3]);
    blue_pat= repmat ([0 1;0 0], [3,3]);
    red_pat= repmat ([0 0;1 0], [3,3]); 

    red_pat(6,:) = []; 
    red_pat(:,6) = [];

    green_pat(6,:) = []; 
    green_pat(:,6) = [];

    blue_pat(6,:) = [];
    blue_pat(:,6) = [];

    X = (im2col(red_pat,[1 1])'.*red_win+im2col(green_pat,[1 1])'.*green_win+im2col(blue_pat,[1 1])'.*blue_win)';

    red_A_GBRG = inv(X'*X)*X'*(red_col');
    blue_A_GBRG = inv(X'*X)*X'*(blue_col');
end
