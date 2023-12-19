% Input Data (soon to be a function)
input_file       = 'IMG_0764-1.png';
input_folder     = 'input/specular3/';
%input_file       = 'IMG_13.jpg';
%input_folder     = 'input/dataset2/';

% input_file       = 'IMG_0004.png';
% input_folder     = 'input/illum/';

%input_file       = '1.png';
%input_folder     = 'input/synthetic/';

%% CAMERA
% 1 : LYTRO1
% 2 : LYTRO2
% 3 : ILLUM
% 4 : SYNTHETIC (WANNER ET AL.)
% 5 : SYNTHETIC (OURS)
% 6 : Jong-Chyi Synthetic
camera   = 3;

%%
% housekeeping
addpath(genpath('required'));
%run('~/Libraries/vl_toolbox/toolbox/vl_setup.m');
[pathstr,name,ext] = fileparts([input_folder '/' input_file]);
if (exist(['output/' name], 'dir') == 0)
    mkdir(['output/' name]);
end
%% REGULARIZER PARAMETERS

reg_generate_all        = 1;
reg_smoothing           = 1;
reg_shading             = 1;
reg_lf_data             = 1;

reg_iteration_thres     = 0.01;


%% INTERNAL PARAMETERS

%%% Pre-processing
brightness          = 1.2;
denoise_radius      = 2  ;

%%% LF sizes                        --------------
UV_radius           = 3                                                   ;
UV_diameter         = (2*UV_radius+1)                                     ;
UV_size             = UV_diameter^2                                       ;

%%% Shearing                        --------------
depth_resolution        = 256                                             ;
alpha_min               = 0.2                                             ;
alpha_max               = 2                                               ;

%%% Analysis                        --------------
% defocus analysis radius
defocus_radius          = 3                                               ;
% correspondence analysis radius
corresp_radius          = UV_radius                                       ;

%%% Regularize                      --------------
CONFI_PENALTY           = 0.6                                             ;
lambda_flat             = 2                                               ;
lambda_smooth           = 2                                               ;
ROBUSTIFY_SMOOTHNESS    = 1                                               ;
gradient_thres          = 1.0                                             ;
SOFTEN_EPSILON          = 1.0                                             ;
CONVERGE_FRACTION       = 1                                               ;
DATA_LAMBDAS            = 1                                               ;
Z_size                  = 0                                               ;
MAX_ITER                = 1                                               ;
WRITE_OUT_FOLDER        = ['output/' name]                                ;

if (camera == 1)
    % LOAD CAMERA DATA
    load('required/camera_data/lytro_2')                                  ;
    image_cords         = image_cords_m                                   ;
    x_size              = size(image_cords_m,2)                           ;
    y_size              = size(image_cords_m,1)                           ;
    % JPEG (RAW IMAGE)
    LF_x_size           = size(image_cords,2)*UV_diameter                 ;
    LF_y_size           = size(image_cords,1)*UV_diameter                 ;
    
    Lytro_RAW           = (imread([input_folder '/' input_file]))         ;
    if (size(Lytro_RAW,3) == 3)
        Lytro_RAW_Demosaic  = im2double(Lytro_RAW)                        ;
    else
        Lytro_RAW_Demosaic  = im2double(demosaic(Lytro_RAW,'bggr'))       ;
    end
end
if (camera == 3)
    %%%%%%%%%% PUT ILLUM CODE HERE
    load('required/camera_data/illum2')                                  ;
    image_cords = round(image_cords);
    x_size              = size(image_cords,2)                           ;
    y_size              = size(image_cords,1)                           ;
    % JPEG (RAW IMAGE)
    LF_x_size           = size(image_cords,2)*UV_diameter                 ;
    LF_y_size           = size(image_cords,1)*UV_diameter                 ;
    
    Lytro_RAW           = (imread([input_folder '/' input_file]))         ;
    Lytro_RAW_Demosaic  = im2double(Lytro_RAW)                        ;
    Lytro_RAW_Demosaic  = Lytro_RAW_Demosaic * 2;
    Lytro_RAW_Demosaic  = min(Lytro_RAW_Demosaic,1);
end
if (camera == 4)
    %%%%%%%%%% PUT SYNTHETIC (WANNER) CODE HERE
end
if (camera == 5)
    %%%%%%%%%% PUT SYNTHETIC (OUR) CODE HERE
end
if (camera == 6)
    %%%%%%%%%% PUT SYNTHETIC (OUR) CODE HERE
    image_cords = zeros(381,330,2);
    for y = 1:381
        for x = 1:330
            x_ind = (x-1)*UV_diameter+round(UV_diameter/2);
            y_ind = (y-1)*UV_diameter+round(UV_diameter/2);
            image_cords(y,x,1) = y_ind;
            image_cords(y,x,2) = x_ind;
        end
    end
    x_size              = size(image_cords,2)                           ;
    y_size              = size(image_cords,1)                           ;
    % JPEG (RAW IMAGE)
    LF_x_size           = size(image_cords,2)*UV_diameter                 ;
    LF_y_size           = size(image_cords,1)*UV_diameter                 ;
    
    Lytro_RAW_Demosaic  = im2double(imread([input_folder '/' input_file]));
end

% GATHER PARAMTERS
LF_parameters       = struct('LF_x_size',LF_x_size,...
    'LF_y_size',LF_y_size,...
    'x_size',x_size,...
    'y_size',y_size,...
    'UV_radius',UV_radius,...
    'UV_diameter',UV_diameter,...
    'UV_size',UV_size,...
    'depth_resolution',depth_resolution,...
    'alpha_min',alpha_min,...
    'alpha_max',alpha_max,...
    'defocus_radius',defocus_radius,...
    'corresp_radius',corresp_radius,...
    'CONFI_PENALTY',CONFI_PENALTY,...
    'DATA_LAMBDAS', DATA_LAMBDAS,...
    'lambda_flat',lambda_flat,...
    'lambda_smooth',lambda_smooth,...
    'ROBUSTIFY_SMOOTHNESS',ROBUSTIFY_SMOOTHNESS,...
    'gradient_thres',gradient_thres,...
    'SOFTEN_EPSILON',SOFTEN_EPSILON,...
    'CONVERGE_FRACTION',CONVERGE_FRACTION,...
    'Z_size',Z_size,...
    'MAX_ITER', MAX_ITER,...
    'WRITE_OUT_FOLDER', WRITE_OUT_FOLDER...
    )                                            ;
% read out file

%% PRE-PROCESSING
% brightness
Lytro_RAW_Demosaic = min(Lytro_RAW_Demosaic*brightness,1)                        ;
% contrast
min_val = min(min(min(Lytro_RAW_Demosaic)));
max_val = max(max(max(Lytro_RAW_Demosaic)));
Lytro_RAW_Demosaic = (Lytro_RAW_Demosaic-min_val)/(max_val-min_val);
% denoise
Lytro_RAW_Demosaic(:,:,1) = wiener2(Lytro_RAW_Demosaic(:,:,1), [denoise_radius denoise_radius]);
Lytro_RAW_Demosaic(:,:,2) = wiener2(Lytro_RAW_Demosaic(:,:,2), [denoise_radius denoise_radius]);
Lytro_RAW_Demosaic(:,:,3) = wiener2(Lytro_RAW_Demosaic(:,:,3), [denoise_radius denoise_radius]);

%% STEP 0: GATHER NECESSARY DATA
fprintf('I. Remapping LF JPEG to our standard                  *******\n');
tic                                                                       ;
% RAW to Remap
LF_Remap            = RAW2REMAP(Lytro_RAW_Demosaic,image_cords,LF_parameters)    ;

IM_Pinhole          = REMAP2PINHOLE(LF_Remap,LF_parameters)                      ;

%% STEP 1: LOCAL ESTIMATION
fprintf('II. Local Estimation (Defocus and Correspondence)     *******\n');
[depth_reg_image_remap,combined_depth,combined_confi,combined_response] = depth_estimation2(LF_Remap, IM_Pinhole, LF_parameters);

%% Focus Remap
LF_Focused_Remap = zeros(LF_y_size,LF_x_size,3);
LF_Remap_Indices = zeros(LF_y_size,LF_x_size,8);
for y = 1:y_size
    for x = 1:x_size
        depth   = depth_reg_image_remap(y,x);
        angular  = get_angular(x,y,depth,LF_Remap, LF_parameters);
        indicies = get_inverse_angular_indices(x,y,depth, LF_parameters);
        
        x_min   = (x-1)*UV_diameter+1;
        x_max   = (x)  *UV_diameter;
        y_min   = (y-1)*UV_diameter+1;
        y_max   = (y)  *UV_diameter;
        
        LF_Focused_Remap(y_min:y_max,x_min:x_max,1) = angular(:,:,1);
        LF_Focused_Remap(y_min:y_max,x_min:x_max,2) = angular(:,:,2);
        LF_Focused_Remap(y_min:y_max,x_min:x_max,3) = angular(:,:,3);
        
        LF_Remap_Indices(y_min:y_max,x_min:x_max,1) = indicies(:,:,1);
        LF_Remap_Indices(y_min:y_max,x_min:x_max,2) = indicies(:,:,2);
        LF_Remap_Indices(y_min:y_max,x_min:x_max,3) = indicies(:,:,3);
        LF_Remap_Indices(y_min:y_max,x_min:x_max,4) = indicies(:,:,4);
        LF_Remap_Indices(y_min:y_max,x_min:x_max,5) = indicies(:,:,5);
        LF_Remap_Indices(y_min:y_max,x_min:x_max,6) = indicies(:,:,6);
        LF_Remap_Indices(y_min:y_max,x_min:x_max,7) = indicies(:,:,7);
        LF_Remap_Indices(y_min:y_max,x_min:x_max,8) = indicies(:,:,8);
    end
end
LF_Remap_Reconstructed  = reconstruct_original(LF_Remap_Indices,LF_Focused_Remap, LF_parameters);
depth_reg_image_remap_up = imresize(depth_reg_image_remap, 7, 'box');
LF_Remap_Depth          = reconstruct_original(LF_Remap_Indices,depth_reg_image_remap_up, LF_parameters);
imwrite(LF_Focused_Remap, [WRITE_OUT_FOLDER '/z_2_regularized_remap.png']);

%% Estimate Color
% STEP 1: Find Ls for each pixel


% chromaticty
% LF_Focused_Remap_Chroma(:,:,1) = LF_Focused_Remap(:,:,1)./(LF_Focused_Remap(:,:,1)+LF_Focused_Remap(:,:,2)+LF_Focused_Remap(:,:,3));
% LF_Focused_Remap_Chroma(:,:,2) = LF_Focused_Remap(:,:,2)./(LF_Focused_Remap(:,:,1)+LF_Focused_Remap(:,:,2)+LF_Focused_Remap(:,:,3));
% LF_Focused_Remap_Chroma(:,:,3) = LF_Focused_Remap(:,:,3)./(LF_Focused_Remap(:,:,1)+LF_Focused_Remap(:,:,2)+LF_Focused_Remap(:,:,3));
% 
% % dark channel
% LF_Focused_Remap_Dark = zeros(LF_y_size,LF_x_size,3);
% for x = 1:LF_x_size
%     for y= 1:LF_y_size
%         min_val = min([LF_Focused_Remap_Chroma(y,x,1);LF_Focused_Remap_Chroma(y,x,2);LF_Focused_Remap_Chroma(y,x,3)]);
%         
%         LF_Focused_Remap_Dark(y,x,1) =  LF_Focused_Remap_Chroma(y,x,1) - min_val;
%         LF_Focused_Remap_Dark(y,x,2) =  LF_Focused_Remap_Chroma(y,x,2) - min_val;
%         LF_Focused_Remap_Dark(y,x,3) =  LF_Focused_Remap_Chroma(y,x,3) - min_val;
%     end
% end

Ls_Estimation = zeros(y_size,x_size,3);
Ls_lower      = zeros(y_size,x_size,3);
Ls_higher     = zeros(y_size,x_size,3);

% estimate color
for y = 1:y_size
    for x = 1:x_size
        x_min   = (x-1)*UV_diameter+1;
        x_max   = (x)  *UV_diameter;
        y_min   = (y-1)*UV_diameter+1;
        y_max   = (y)  *UV_diameter;
        
        % SLOPE ESTIMATION: kmeans cluster
        R = LF_Focused_Remap(y_min:y_max,x_min:x_max,1);
        G = LF_Focused_Remap(y_min:y_max,x_min:x_max,2);
        B = LF_Focused_Remap(y_min:y_max,x_min:x_max,3);
        
        R = R(:);
        G = G(:);
        B = B(:);
        
        RGB_data = [R G B];
        
        % find slope
        % kmeans
        % [idx,C] = kmeans(RGB_data,2);
        
        % minimum
        C(1,1) = min(R);
        C(1,2) = min(G);
        C(1,3) = min(B);
        
        C(2,1) = max(R);
        C(2,2) = max(G);
        C(2,3) = max(B);
        
        Ls_Estimation(y,x,1) = C(2,1)-C(1,1);
        Ls_Estimation(y,x,2) = C(2,2)-C(1,2);
        Ls_Estimation(y,x,3) = C(2,3)-C(1,3);
        
        Ls_lower(y,x,1) = C(1,1);
        Ls_lower(y,x,2) = C(1,2);
        Ls_lower(y,x,3) = C(1,3);
        
        Ls_higher(y,x,1) = C(2,1);
        Ls_higher(y,x,2) = C(2,2);
        Ls_higher(y,x,3) = C(2,3);
        
    end
end

%% Estimate Ls

% magnitude of Ls_Estimation
Ls_Estimation_magnitude     = sqrt((Ls_Estimation(:,:,1).^2+Ls_Estimation(:,:,2).^2+Ls_Estimation(:,:,3).^2)/3);

% chromaticty of Ls_Estimation
Ls_Estimation_chroma        = Ls_Estimation;
Ls_Estimation_chroma(:,:,1) = Ls_Estimation(:,:,1)./(Ls_Estimation(:,:,1)+Ls_Estimation(:,:,2)+Ls_Estimation(:,:,3));
Ls_Estimation_chroma(:,:,2) = Ls_Estimation(:,:,2)./(Ls_Estimation(:,:,1)+Ls_Estimation(:,:,2)+Ls_Estimation(:,:,3));
Ls_Estimation_chroma(:,:,3) = Ls_Estimation(:,:,3)./(Ls_Estimation(:,:,1)+Ls_Estimation(:,:,2)+Ls_Estimation(:,:,3));

% Estimation
% parameters
brightness_constraint = 2;
light_source_colors   = 2;
light_source_similar  = 0.03;

Ls_Estimation_thres = quantile(Ls_Estimation_magnitude(:),100/brightness_constraint);
Ls_Estimation_thres = Ls_Estimation_thres(100/brightness_constraint);

R_estimate = [];
G_estimate = [];
B_estimate = [];

for x = 1:x_size
    for y = 1:y_size
        if (Ls_Estimation_magnitude(y,x) > Ls_Estimation_thres)
            R_estimate = [R_estimate;Ls_Estimation_chroma(y,x,1)];
            G_estimate = [G_estimate;Ls_Estimation_chroma(y,x,2)];
            B_estimate = [B_estimate;Ls_Estimation_chroma(y,x,3)];
        end
    end
end

RGB_estimate_data = [R_estimate G_estimate B_estimate];
[ind,estimated_colors] = kmeans(RGB_estimate_data,light_source_colors);

% OUR ESTIMATED Ls COLORS
estimated_colors;

%% Estimate ones with Ls
% 1. pixels with large variance
%h = fspecial('average', [UV_diameter*2+1 UV_diameter*2+1]);
LF_Focused_Remap_mean = LF_Focused_Remap;
%imfilter(LF_Focused_Remap,h,'symmetric');


percent_cut = 0.2;

for y = 1:y_size
    for x = 1:x_size
        x_min   = (x-1)*UV_diameter+1;
        x_max   = (x)  *UV_diameter;
        y_min   = (y-1)*UV_diameter+1;
        y_max   = (y)  *UV_diameter;
        
        R_vec = LF_Focused_Remap(y_min:y_max,x_min:x_max,1);
        R_vec = R_vec(:);
        
        G_vec = LF_Focused_Remap(y_min:y_max,x_min:x_max,2);
        G_vec = G_vec(:);
        
        B_vec = LF_Focused_Remap(y_min:y_max,x_min:x_max,3);
        B_vec = B_vec(:);
        
        LF_Focused_Remap_mean(y_min:y_max,x_min:x_max,1) = quantile(R_vec,percent_cut);
        LF_Focused_Remap_mean(y_min:y_max,x_min:x_max,2) = quantile(G_vec,percent_cut);
        LF_Focused_Remap_mean(y_min:y_max,x_min:x_max,3) = quantile(B_vec,percent_cut);
    end
end

LF_Focused_Remap_var  = LF_Focused_Remap - LF_Focused_Remap_mean;
LF_Focused_Remap_var_abs = sqrt((LF_Focused_Remap_var(:,:,1).^2+LF_Focused_Remap_var(:,:,2).^2+LF_Focused_Remap_var(:,:,3).^2)/3);
LF_Focused_Remap_var  = abs(LF_Focused_Remap_var);


specular_confidence     = zeros(y_size,x_size);
specular_confidence_RGB = zeros(y_size,x_size,3);

for y = 1:y_size
    for x = 1:x_size
        x_min   = (x-1)*UV_diameter+1;
        x_max   = (x)  *UV_diameter;
        y_min   = (y-1)*UV_diameter+1;
        y_max   = (y)  *UV_diameter;
        
        % max 
        specular_confidence(y,x) = max(max(LF_Focused_Remap_var_abs(y_min:y_max,x_min:x_max)));
        specular_confidence_RGB(y,x,1) = max(max(LF_Focused_Remap_var(y_min:y_max,x_min:x_max,1)));
        specular_confidence_RGB(y,x,2) = max(max(LF_Focused_Remap_var(y_min:y_max,x_min:x_max,2)));
        specular_confidence_RGB(y,x,3) = max(max(LF_Focused_Remap_var(y_min:y_max,x_min:x_max,3)));
    end
end

% 2. similar color to ls
LF_Focused_Remap_Color_Similar = zeros(y_size,x_size);
for Ls = 1:light_source_colors
    LF_Focused_Remap_Color_Similar_Tmp = roicolor(Ls_Estimation_chroma(:,:,1),estimated_colors(Ls,1)-light_source_similar,estimated_colors(Ls,1)+light_source_similar);
    LF_Focused_Remap_Color_Similar_Tmp = LF_Focused_Remap_Color_Similar_Tmp & roicolor(Ls_Estimation_chroma(:,:,2),estimated_colors(Ls,2)-light_source_similar,estimated_colors(Ls,2)+light_source_similar);
    LF_Focused_Remap_Color_Similar_Tmp = LF_Focused_Remap_Color_Similar_Tmp & roicolor(Ls_Estimation_chroma(:,:,3),estimated_colors(Ls,3)-light_source_similar,estimated_colors(Ls,3)+light_source_similar);
    LF_Focused_Remap_Color_Similar     = LF_Focused_Remap_Color_Similar + LF_Focused_Remap_Color_Similar_Tmp;
end
imshow(double(LF_Focused_Remap_Color_Similar));

% 3. combine estimate variance and similar color to Ls
Ls_Focused_Remap_Confidence = zeros(LF_y_size,LF_x_size);
for y = 1:y_size
    for x = 1:x_size
        x_min   = (x-1)*UV_diameter+1;
        x_max   = (x)  *UV_diameter;
        y_min   = (y-1)*UV_diameter+1;
        y_max   = (y)  *UV_diameter;
        
        Ls_Focused_Remap_Confidence(y_min:y_max,x_min:x_max) =LF_Focused_Remap_var_abs(y_min:y_max,x_min:x_max).*LF_Focused_Remap_Color_Similar(y,x,1);
    end
end

%% Remove Specularities Using Bilateral Fitler

LF_Focused_Diffuse = LF_Focused_Remap;
Ls_Focused_Remap_Confidence = LF_Focused_Remap_var_abs;

% parameter
capping = 0.4;
Ls_Focused_Remap_Confidence_Scaled = min(Ls_Focused_Remap_Confidence,capping)/capping;

% remove specularities
for y = 1:y_size
    for x = 1:x_size
        x_min   = (x-1)*UV_diameter+1;
        x_max   = (x)  *UV_diameter;
        y_min   = (y-1)*UV_diameter+1;
        y_max   = (y)  *UV_diameter;
        
        R = 0;
        G = 0;
        B = 0;
        norm_weight = 0;
        
        for i = x_min:x_max
            for j = y_min:y_max
                R = R + (1-Ls_Focused_Remap_Confidence_Scaled(j,i))*LF_Focused_Remap(j,i,1);
                G = G + (1-Ls_Focused_Remap_Confidence_Scaled(j,i))*LF_Focused_Remap(j,i,2);
                B = B + (1-Ls_Focused_Remap_Confidence_Scaled(j,i))*LF_Focused_Remap(j,i,3);
                norm_weight = norm_weight + (1-Ls_Focused_Remap_Confidence_Scaled(j,i));
            end
        end
        
        R = R/norm_weight;
        G = G/norm_weight;
        B = B/norm_weight;
        
        % reduce angular
        for i = x_min:x_max
            for j = y_min:y_max
                LF_Focused_Diffuse(j,i,1) = LF_Focused_Remap(j,i,1)*(1-Ls_Focused_Remap_Confidence_Scaled(j,i))+R*(Ls_Focused_Remap_Confidence(j,i));
                LF_Focused_Diffuse(j,i,2) = LF_Focused_Remap(j,i,2)*(1-Ls_Focused_Remap_Confidence_Scaled(j,i))+G*(Ls_Focused_Remap_Confidence(j,i));
                LF_Focused_Diffuse(j,i,3) = LF_Focused_Remap(j,i,3)*(1-Ls_Focused_Remap_Confidence_Scaled(j,i))+B*(Ls_Focused_Remap_Confidence(j,i));
            end
        end
    end
end
%% Transfer back to original Space
imshow([LF_Focused_Remap LF_Focused_Diffuse abs(LF_Focused_Remap-LF_Focused_Diffuse)*1.5]);
LF_Remap_Reconstructed  = reconstruct_original(LF_Remap_Indices,LF_Focused_Remap, LF_parameters);
LF_Focused_Remap_var_abs_Reconstructed  = reconstruct_original(LF_Remap_Indices,LF_Focused_Remap_var_abs, LF_parameters);
LF_Diffuse_Reconstructed  = reconstruct_original(LF_Remap_Indices,LF_Focused_Diffuse, LF_parameters);
imshow([LF_Remap_Reconstructed LF_Diffuse_Reconstructed abs(LF_Diffuse_Reconstructed-LF_Remap_Reconstructed)*1.5]);

IM_Original_Refocus = REMAP2REFOCUS_SIMPLE(LF_Remap_Reconstructed);
IM_Diffuse_Refocus = REMAP2REFOCUS_SIMPLE(LF_Diffuse_Reconstructed);
IM_Specular_Refocus = REMAP2REFOCUS_SIMPLE(abs(LF_Diffuse_Reconstructed-LF_Remap_Reconstructed)*1.5);

% 
% neighbor_radius = 5;
% 
% IM_Diffuse_Refocus_Spatial = IM_Diffuse_Refocus;
% for y = 1:y_size
%     for x = 1:x_size
%         center_R = IM_Original_Refocus(y,x,1);
%         center_G = IM_Original_Refocus(y,x,2);
%         center_B = IM_Original_Refocus(y,x,3);
%         
%         x_min = max(x-neighbor_radius,1);
%         y_min = max(y-neighbor_radius,1);
%         x_max = min(x+neighbor_radius,x_size);
%         y_max = min(y+neighbor_radius,y_size);
%         
%         sum_val = [0 0 0];
%         weight = [0 0 0];
%         for i = x_min:x_max
%              for j = y_min:y_max
%                  target_R = IM_Original_Refocus(j,i,1);
%                  target_G = IM_Original_Refocus(j,i,2);
%                  target_B = IM_Original_Refocus(j,i,3);
%                  diff = sqrt(sum(([center_R center_G center_B]-[target_R target_G target_B]).^2)/3);
%                  
%                  if (diff < 0.05)
%                      sum_val(1) = sum_val(1) + IM_Diffuse_Refocus(j,i,1);
%                      sum_val(2) = sum_val(2) + IM_Diffuse_Refocus(j,i,2);
%                      sum_val(3) = sum_val(3) + IM_Diffuse_Refocus(j,i,3);
%                      
%                      weight(1) = weight(1) + abs(IM_Original_Refocus(j,i,1)-IM_Diffuse_Refocus(j,i,1));
%                      weight(2) = weight(2) + abs(IM_Original_Refocus(j,i,2)-IM_Diffuse_Refocus(j,i,2));
%                      weight(3) = weight(3) + abs(IM_Original_Refocus(j,i,3)-IM_Diffuse_Refocus(j,i,3));
%                  end
%              end
%         end
%         if weight > 0
%             
%         IM_Diffuse_Refocus_Spatial(y,x,1) = sum_val(1)/weight(1);
%         IM_Diffuse_Refocus_Spatial(y,x,2) = sum_val(2)/weight(2);
%         IM_Diffuse_Refocus_Spatial(y,x,3) = sum_val(3)/weight(3);
%         end
%         
%     end
% end
%     
% 
% radius = 5;
% 
% IM_Diffuse_Refocus_Spatial = IM_Diffuse_Refocus;
% 
% for y = 1:y_size
%     for x = 1:x_size
%         x_min = max(x-neighbor_radius,1);
%         y_min = max(y-neighbor_radius,1);
%         x_max = min(x+neighbor_radius,x_size);
%         y_max = min(y+neighbor_radius,y_size);
%         
%         R = 0;
%         G = 0;
%         B = 0;
%         norm_weight = [0;0;0];
%         
%         for i = x_min:x_max
%             for j = y_min:y_max
%                 R = R + (1-IM_Specular_Refocus(j,i,1))*IM_Diffuse_Refocus(j,i,1);
%                 G = G + (1-IM_Specular_Refocus(j,i,2))*IM_Diffuse_Refocus(j,i,2);
%                 B = B + (1-IM_Specular_Refocus(j,i,3))*IM_Diffuse_Refocus(j,i,3);
%                 norm_weight(1) = norm_weight(1) + (1-IM_Specular_Refocus(j,i,1));
%                 norm_weight(2) = norm_weight(2) + (1-IM_Specular_Refocus(j,i,2));
%                 norm_weight(3) = norm_weight(3) + (1-IM_Specular_Refocus(j,i,3));
%             end
%         end
%         
%         R = R/norm_weight(1);
%         G = G/norm_weight(2);
%         B = B/norm_weight(3);
%         
%         reduce angular
%         for i = x_min:x_max
%             for j = y_min:y_max
%                 IM_Diffuse_Refocus_Spatial(j,i,1) = IM_Diffuse_Refocus(j,i,1)*(1-IM_Specular_Refocus(j,i,1))+R*(IM_Specular_Refocus(j,i,1));
%                 IM_Diffuse_Refocus_Spatial(j,i,2) = IM_Diffuse_Refocus(j,i,2)*(1-IM_Specular_Refocus(j,i,2))+G*(IM_Specular_Refocus(j,i,2));
%                 IM_Diffuse_Refocus_Spatial(j,i,3) = IM_Diffuse_Refocus(j,i,3)*(1-IM_Specular_Refocus(j,i,3))+B*(IM_Specular_Refocus(j,i,3));
%             end
%         end
%     end
% end

% % IM_Focused_Diffuse_neighbor = IM_Diffuse_Refocus;
% % % remove neighborhood
% neighbor_radius = 20;
% for y = 1:y_size
%     for x = 1:x_size
%         
%         % target before
%         target_before(1) = IM_Original_Refocus(y,x,1);
%         target_before(2) = IM_Original_Refocus(y,x,2);
%         target_before(3) = IM_Original_Refocus(y,x,3);
%         % target after
%         target_after(1) = IM_Diffuse_Refocus(y,x,1);
%         target_after(2) = IM_Diffuse_Refocus(y,x,2);
%         target_after(3) = IM_Diffuse_Refocus(y,x,3);
%         
%         diff = sqrt(sum((target_before-target_after).^2)/3);
%         if (diff > 0.02)
%         i_min = max(x-neighbor_radius,1);
%         j_min = max(y-neighbor_radius,1);
%         i_max = min(x+neighbor_radius,x_size);
%         j_max = min(y+neighbor_radius,y_size);
%         for i = i_min:i_max
%             for j = j_min:j_max
%                 target(1) = IM_Diffuse_Refocus(j,i,1);
%                 target(2) = IM_Diffuse_Refocus(j,i,2);
%                 target(3) = IM_Diffuse_Refocus(j,i,3);
%                 diff = sqrt(sum((target_before-target).^2)/3);
%                 if (diff < 0.1)
%                     IM_Focused_Diffuse_neighbor(j,i,1) = target_after(1);
%                     IM_Focused_Diffuse_neighbor(j,i,2) = target_after(2);
%                     IM_Focused_Diffuse_neighbor(j,i,3) = target_after(3);
%                 end
%             end
%         end
%         end
%     end
% end


%% STEP 3: LOCAL ESTIMATION 2
fprintf('III. Local Estimation (Defocus and Correspondence)     *******\n');

% scale for confidence
%specular_confidence_scaled = (1-double(specular_confidence>0.2));
%imshow(specular_confidence_scaled);
specular_confidence_scaled = (1-double(specular_confidence)).^5;
specular_confidence_RGB_scaled = (1-double(specular_confidence_RGB)).^5;

%specular_confidence_scaled = 1-IM_Specular_Mask;
% call

%[specular_response] = compute_LFdepth_Specular(LF_Remap, IM_Pinhole, Ls_Estimation_chroma, LF_parameters);
%[specular_response] = compute_LFdepth_Specular(LF_Remap, IM_Pinhole, Ls_Estimation_chroma, LF_parameters);
[depth_reg_image_remap_FINAL_specular depth_specular_unregularized depth_specular_confidence_FINAL] = depth_estimation_specular(LF_Remap, IM_Pinhole,LF_parameters,combined_depth,combined_confi,specular_confidence_scaled,Ls_Estimation_chroma,combined_response);
[depth_reg_image_remap_FINAL depth_regular_unregularized depth_regular_confidence_FINAL z] = depth_estimation(LF_Remap, IM_Pinhole,LF_parameters,combined_depth,combined_confi,specular_confidence_scaled);

%final_confidence = depth_regular_confidence_FINAL;
%final_confidence = specular_confidence_scaled.*depth_regular_confidence_FINAL + (1-specular_confidence_scaled).*depth_specular_confidence_FINAL;
%final_depth      = depth_regular_unregularized.*specular_confidence_scaled + (1-specular_confidence_scaled).*depth_specular_unregularized;
%final_depth      = depth_regular_unregularized;

matte_buffer = IM_Specular_Refocus;
matte_buffer = double(matte_buffer > 0.2);
h = fspecial('gaussian', [3 3], 1);
matte_buffer = imfilter(matte_buffer,h,'symmetric');
matte_buffer = 1-matte_buffer;

mask = matte_buffer;
mask = sqrt((mask(:,:,1).^2+mask(:,:,2).^2+mask(:,:,3).^2)/3);
mask = 1-mask;

final_depth = mask.*depth_specular_unregularized/256+(1-mask).*depth_regular_unregularized/256;
final_confidence      = mask.*depth_specular_confidence_FINAL+(1-mask).*depth_regular_confidence_FINAL;


%[final_reg_depth] = regularize_depth_synth(IM_Pinhole,final_depth, final_confidence, LF_parameters);


[final_reg_depth] = regularize_depth(IM_Pinhole,final_depth, final_confidence, LF_parameters);


% write out
if (1)
    imwrite(depth_reg_image_remap/256, [WRITE_OUT_FOLDER '/zz1_initial.png']);
    
    imwrite(depth_reg_image_remap_FINAL/256, [WRITE_OUT_FOLDER '/zz1b1_DIFFUSE.png']);

    imwrite(depth_reg_image_remap_FINAL_specular/256, [WRITE_OUT_FOLDER '/zz1b2_SPECULAR.png']);
    
    imwrite(depth_regular_unregularized/256, [WRITE_OUT_FOLDER '/zz1c1_DIFFUSE_unreg.png']);
    
    imwrite(depth_specular_unregularized/256, [WRITE_OUT_FOLDER '/zz1c2_SPECULAR_unreg.png']);
    
    imwrite(combined_confi, [WRITE_OUT_FOLDER '/zz1d1_initial_confidence.png']);
    
    imwrite(depth_regular_confidence_FINAL, [WRITE_OUT_FOLDER '/zz1d2_DIFFUSE_confidence.png']);
    
    imwrite(depth_specular_confidence_FINAL, [WRITE_OUT_FOLDER '/zz1d3_SPECULAR_confidence.png']);
    
    imwrite(final_confidence, [WRITE_OUT_FOLDER '/zz1d4_FINAL_confidence.png']);
    
    imwrite(final_depth/256, [WRITE_OUT_FOLDER '/zz1c3_FINAL_unreg.png']);
    
    imwrite(1-final_reg_depth, [WRITE_OUT_FOLDER '/zz1b3_FINAL.png']);
    
    imwrite(depth_reg_image_remap_FINAL_specular/256, [WRITE_OUT_FOLDER '/zz1b2_SPECULAR.png']);
    
    imwrite(depth_regular_unregularized/256, [WRITE_OUT_FOLDER '/zz1c1_DIFFUSE_unreg.png']);
    
    imwrite(depth_specular_unregularized/256, [WRITE_OUT_FOLDER '/zz1c2_SPECULAR_unreg.png']);
    
    imwrite(specular_confidence_scaled, [WRITE_OUT_FOLDER '/zzzz1d_confidence_specular.png']);
    imwrite(specular_confidence_RGB_scaled, [WRITE_OUT_FOLDER '/zzzz1d2_confidence_specular_rgb.png']);
    
    imwrite(IM_Original_Refocus, [WRITE_OUT_FOLDER '/zzzz2a_original.png']);
    imwrite(IM_Diffuse_Refocus, [WRITE_OUT_FOLDER '/zzzz2b_diffuse.png']);
    imwrite(IM_Specular_Refocus, [WRITE_OUT_FOLDER '/zzzz2c_specular.png']);
    imwrite(LF_Remap_Reconstructed, [WRITE_OUT_FOLDER '/zzzz3a_original_remap.png']);
    imwrite(LF_Diffuse_Reconstructed, [WRITE_OUT_FOLDER '/zzzz3b_diffuse_remap.png']);
    imwrite(abs(LF_Diffuse_Reconstructed-LF_Remap_Reconstructed)*1.5, [WRITE_OUT_FOLDER '/zzzz3c_specular_remap.png']);
    imwrite(LF_Focused_Remap, [WRITE_OUT_FOLDER '/zzzz4a_original_consistency_remap.png']);
    imwrite(LF_Focused_Diffuse, [WRITE_OUT_FOLDER '/zzzz4b_diffuse_consistency_remap.png']);
    imwrite(abs(LF_Focused_Remap-LF_Focused_Diffuse)*1.5, [WRITE_OUT_FOLDER '/zzzz4c_specular_consistency_remap.png']);
    
    imwrite(Ls_Estimation, [WRITE_OUT_FOLDER '/zzzz5_ls_estimation.png']);
    imwrite(LF_Focused_Remap_var_abs, [WRITE_OUT_FOLDER '/zzzz6a_variance_confidence.png']);
    imwrite(LF_Focused_Remap_Color_Similar, [WRITE_OUT_FOLDER '/zzzz6b_color_confidence.png']);
    imwrite(Ls_Focused_Remap_Confidence, [WRITE_OUT_FOLDER '/zzzz6c_combined_variacne_color_confidence.png']);
    
    
    color_img = zeros(1,light_source_colors,3);
    for ind = 1:light_source_colors
        color_img(1,ind,1) = estimated_colors(ind,1);
        color_img(1,ind,2) = estimated_colors(ind,2);
        color_img(1,ind,3) = estimated_colors(ind,3);
    end
    imwrite(color_img, [WRITE_OUT_FOLDER '/zzzz7_estimated_colors.png']);
end