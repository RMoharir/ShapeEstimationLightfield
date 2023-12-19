function [depth_reg_image_remap,combined_depth,combined_confi] = depth_estimation_specular(LF_Remap,IM_Pinhole,LF_parameters,combined_depth_p,combined_confi_p,specular_confidence_map,Ls_Estimation_chroma,combined_response)
%DEPTH_ESTIMATION Summary of this function goes here
%   Detailed explanation goes here

%%%%%%%%%%%%%%%% CONFIDENCE MEASUREMENT

%%%% iii. depth estimation : eqn. (6)
%defocus_depth = DEPTH_ESTIMATION(defocus_response,1)                      ;
%corresp_depth = DEPTH_ESTIMATION(corresp_response,0)                      ;

%%%% iv. confidence measure: eqn. (7)

    [specular_response] = compute_LFdepth_Specular(LF_Remap, IM_Pinhole, Ls_Estimation_chroma, LF_parameters);
    specular_confi = ALM_CONFIDENCE(specular_response,IM_Pinhole,LF_parameters,0);
    [specular_confi,combined_confi] = NORMALIZE_CONFIDENCE(specular_confi,combined_confi_p);
    
    specular_confi = specular_confi/max(max(specular_confi));
    
    combined_depth = (DEPTH_ESTIMATION(specular_response,0));
    
    combined_confi = specular_confi;
    combined_confi(isnan(combined_confi)==1)=0;
    
% %%%%%%%% REGULARIZATION
%% Regularize                      --------------
LF_parameters.CONFI_PENALTY           = 0.6                                             ;
LF_parameters.lambda_flat             = 2                                               ;%2
LF_parameters.lambda_smooth           = 1                                               ;%1
LF_parameters.ROBUSTIFY_SMOOTHNESS    = 1                                               ;
LF_parameters.gradient_thres          = 1.0                                             ;
LF_parameters.SOFTEN_EPSILON          = 0.001                                           ;
LF_parameters.CONVERGE_FRACTION       = 0.0005                                          ;
LF_parameters.DATA_LAMBDAS            = 1                                               ;
LF_parameters.MAX_ITER                = 20                                               ;

CONFI_PENALTY = LF_parameters.CONFI_PENALTY ;
lambda_flat = LF_parameters.lambda_flat;
lambda_smooth = LF_parameters.lambda_smooth;
ROBUSTIFY_SMOOTHNESS = LF_parameters.ROBUSTIFY_SMOOTHNESS;
gradient_thres = LF_parameters.gradient_thres;
SOFTEN_EPSILON = LF_parameters.SOFTEN_EPSILON;
CONVERGE_FRACTION = LF_parameters.CONVERGE_FRACTION;
DATA_LAMBDAS = LF_parameters.DATA_LAMBDAS;
MAX_ITER = LF_parameters.MAX_ITER;

data_term  = {double(combined_depth) * -4}                                              ;
conf_term  = {combined_confi.^LF_parameters.CONFI_PENALTY}                                            ;

[A_d, b_d] = reg_data_term_f(data_term, conf_term, LF_parameters);
[A_s, b_s] = reg_smooth_term_f(IM_Pinhole, [0 -1 0; -1 4 -1; 0 -1 0], LF_parameters);
[A_f1, b_f1] = reg_smooth_term_f(IM_Pinhole, [1 -1]/2, LF_parameters);
[A_f2, b_f2] = reg_smooth_term_f(IM_Pinhole, ([1 -1]/2)', LF_parameters);

A = [A_d; lambda_flat * A_f1; lambda_flat * A_f2;lambda_smooth * A_s];
b = [b_d; lambda_flat * b_f1; lambda_flat * b_f2; lambda_smooth * b_s];

% solve for equation
depth_reg_image = solve_lqn(A,b,LF_parameters);
depth_reg_image_remap = round(depth_reg_image/-4);

% depth_output  = DEPTH_MRF(combined_depth,combined_confi,IM_Pinhole,LF_parameters)

depth_reg_image_viz = mat2gray(depth_reg_image);
visualizeZ_3D(depth_reg_image_viz*256,IM_Pinhole);

% depth_reg_image_remap = DEPTH_MRF2(combined_depth,combined_confi,IM_Pinhole,LF_parameters);

% if (1)
%     defocus_depth = DEPTH_ESTIMATION(defocus_response,0)                      ;
%     corresp_depth = DEPTH_ESTIMATION(corresp_response,0)                      ;
%     imwrite(defocus_depth/256, [WRITE_OUT_FOLDER '/0a_defocus_depth_smooth.png']);
%     imwrite(corresp_depth/256, [WRITE_OUT_FOLDER '/0a_corresp_depth_smooth.png']);
%     imwrite(defocus_confi, [WRITE_OUT_FOLDER '/0c_defocus_confi.png']);
%     imwrite(corresp_confi, [WRITE_OUT_FOLDER '/0d_corresp_confi.png']);
%     imwrite(combined_depth/256, [WRITE_OUT_FOLDER '/1a_combined_depth.png']);
%     imwrite(combined_confi, [WRITE_OUT_FOLDER '/1b_combined_confi.png']);
%     imwrite(depth_reg_image, [WRITE_OUT_FOLDER '/2_f_combined_depth_smooth.png']);
% end

end

