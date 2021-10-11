%%% A script demonstrating how to use the proposed method to denoise and estimate DKI tensors 
% -----------------------------------------------------------------------------------
%%% Please cite: 
%%% Li Guo, Jian Lyu, Zhe Zhang, Jinping Shi, Qianjin Feng, Yanqiu Feng, Mingyong Gao, and Xinyuan Zhang
%%% "A Joint Framework for Denoising and Estimating Diffusion Kurtosis Tensors Using Multiple Prior Information",
%%% IEEE TMI,2021,DOI:10.1109/TMI.2021.3112515
%------------------------------------------------------------------------------------
%%% Date: 09-16-2021
clc;clear;close all;
addpath('./utiles')

slice = 19;
Ncoils = 1;

%%
temp1 = 1;
temp2 = 2*Ncoils - 1;
while (temp2 > 1)
    temp1 = temp1 * temp2;
    temp2 = temp2 - 2;
end
temp3 = (2.^(Ncoils-1)) * factorial(Ncoils-1);
temp4 = sqrt(pi/2)*temp1/temp3;

%% load data   
load('./data/data_Meningiomas.mat');

%%  
im_r = abs(squeeze(double(img(:,:,slice,:))));
[nx ny ndwi] = size(im_r);
clear img
 
%%
a=im_r(:,:,1);
S0_max = max(max(a(Mask)));% max(S0);
im_r = im_r/S0_max;
im_r = im_r*sqrt(8);

%% bmat
normbvec = sqrt(sum(bvec(1:3, :).^2, 1)); 
normbvec(normbvec == 0) = 1;   
bvec(1:3, :) = bvec(1:3, :)./repmat(normbvec, [3 1]);     
bvec(isnan(bvec)) = 0;

bval = bval'*1e-3; 
bvec = bvec';   
grad = bvec;
grad(:,4) = bval;
normgrad = sqrt(sum(grad(:, 1:3).^2, 2)); 
normgrad(normgrad == 0) = 1;   
grad(:, 1:3) = grad(:, 1:3)./repmat(normgrad, [1 3]);     
grad(isnan(grad)) = 0;
[D_ind, D_cnt] = createTensorOrder(2);  
[W_ind, W_cnt] = createTensorOrder(4);    
bmat = [ones([ndwi,1]), -repmat(bval,[1,6]).*bvec(:,D_ind(:,1)).*bvec(:,D_ind(:,2))*diag(D_cnt), (1/6)*repmat(bval,[1,15]).^2.*bvec(:,W_ind(:,1)).*bvec(:,W_ind(:,2)).*bvec(:,W_ind(:,3)).*bvec(:,W_ind(:,4))*diag(W_cnt)];
nparam = size(bmat,2);
bvecmat = [bvec(:,D_ind(:,1)).*bvec(:,D_ind(:,2))*diag(D_cnt), bvec(:,W_ind(:,1)).*bvec(:,W_ind(:,2)).*bvec(:,W_ind(:,3)).*bvec(:,W_ind(:,4))*diag(W_cnt)];

%%
im_r(im_r<=0)=eps;
S = reshape(im_r,[nx,ny,1,ndwi]);
param.xstart = 5;param.xlength = 13;param.ystart = 86;param.ylength = 14;
sigma_map = cal_sigma_bacg(S,Ncoils,grad,param);
sigma_map = repmat(sigma_map,[nx ny]);

%% initialization estimation      
[~,~,dt_22_initial] = dki_fit(S, grad, Mask, [1 1 1], [], 3.5);
clear S

%%  parameter setting 
t = 5;%radio of search window 
f = 2;%radio of patch 
beta = 1.5;%smoothing parameter
alpha = 0.015;
lambda = alpha^2;%TV weight

%% Preprocess data by removing the backgrounds to accelerate computation
M_col=sum(Mask,1);
pp=find(M_col~=0); 
ind_ymin=pp(1)-t;
ind_ymax=pp(end)+t;
M_row=sum(Mask,2);
pp=find(M_row~=0); 
ind_xmin=pp(1)-t;
ind_xmax=pp(end)+t;

im_r=im_r(ind_xmin:ind_xmax,ind_ymin:ind_ymax,:);
[nx ny ndwi] = size(im_r);
Mask=Mask(ind_xmin:ind_xmax,ind_ymin:ind_ymax);
sigma_map=sigma_map(ind_xmin:ind_xmax,ind_ymin:ind_ymax).*Mask;
dt_22_initial=dt_22_initial(ind_xmin:ind_xmax,ind_ymin:ind_ymax,:);
x_initial = reshape(dt_22_initial,[nx*ny 22]); 

x_initial(:,23) = reshape(sigma_map,[nx*ny,1]);

%% optimization of the proposed M1NCM-NSS-LSS-PR method
options = struct('MaxIter',601,'GradObj','on','Display','iter','LargeScale','off','HessUpdate','lbfgs','InitialHessType','identity','GoalsExactAchieve',0);

h = beta*sigma_map;
nss_weight = fun_weight_nss(im_r,t,f,h);    
nss_weight(isnan(nss_weight))=eps;nss_weight(isinf(nss_weight))=eps;

[im_r_w im_r_w_square]=fun_objective_factor(nss_weight,im_r,t,Mask);
im_r_w_square = im_r_w_square.*repmat(Mask,[1 1 ndwi]);   
im_r_w_square = sum(im_r_w_square,3); 
clear nss_weight
     
tic            
[dt_23,fval_total_obj]=  fminlbfgs(@(x)fun_m1ncmnsslsspr_nonstationary(x,bvecmat,bmat,max(bval),im_r_w,im_r_w_square,Ncoils,Mask,lambda,temp4),x_initial,options);
time_fitting = toc
 
   
%% scalar maps               
dt_23_ = reshape(dt_23,[nx,ny,23]);       
x_sigma = dt_23_(:,:,23);
dt_22 = dt_23_(:,:,1:22);
b0 = exp(dt_22(:,:,1));
dt_21 = dt_22(:,:,2:22);     
D_apprSq = 1./(sum(dt_21(:,:,[1 4 6]),3)/3).^2;                    
dt_21(:,:,7:21) = dt_21(:,:,7:21) .* repmat(D_apprSq,[1 1 15]);      
dt_21_ = reshape(dt_21,nx,ny,1,21);
[fa,md,mk] = dki_parameters(dt_21_, Mask);
 
  

    


