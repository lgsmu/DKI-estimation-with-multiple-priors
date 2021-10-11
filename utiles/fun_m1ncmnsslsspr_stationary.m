function [feval grad] = fun_m1ncmnsslsspr_stationary(x,bvecmat,bmat,max_bval,im_r_w,im_r_w_square,sigma_map,Ncoils,mask,lambda,temp4)

%%% A script demonstrating how to use the proposed method to denoise and estimate DKI tensors 
% -----------------------------------------------------------------------------------
%%% Please cite: 
%%% Li Guo, Jian Lyu, Zhe Zhang, Jinping Shi, Qianjin Feng, Yanqiu Feng, Mingyong Gao, and Xinyuan Zhang
%%% "A Joint Framework for Denoising and Estimating Diffusion Kurtosis Tensors Using Multiple Prior Information",
%%% IEEE TMI,2021,DOI:10.1109/TMI.2021.3112515
%------------------------------------------------------------------------------------
%%% Date: 09-16-2021

load('pp_initial.mat')

%% data fidelity term and physical relevance term
[m n ndwi] = size(im_r_w);
sigma_map = repmat(sigma_map,[1 1 ndwi]);

xp = reshape(x,[m n 22]);clear x
bmatp = repmat(reshape(bmat,[1 1 ndwi 22]),[m n 1 1]);clear bmat
bvecmatp = repmat(reshape(bvecmat,[1 1 ndwi 21]),[m n 1 1]);clear bvecmat
s = zeros([m n ndwi]);
s_pos = s;
grad_3_kurtosis = zeros(m,n,15);
grad_3_diffusion = zeros(m,n,6);
grad_3_constraint3 = zeros(m,n,15);
for ii=1:m
    for jj=1:n      
        if mask(ii,jj)>0
            s(ii,jj,:) = exp(squeeze(bmatp(ii,jj,:,:))*squeeze(xp(ii,jj,:)));  
            
            diffusion = squeeze(bvecmatp(ii,jj,:,1:6))*squeeze(xp(ii,jj,2:7));
            kurtosis = squeeze(bvecmatp(ii,jj,:,7:21))*squeeze(xp(ii,jj,8:22));
            constraint3 = 3*diffusion/max_bval-kurtosis;
            kurtosis_pos = exp(-kurtosis/0.01);
            constraint3_pos = exp(-constraint3/0.01);
            s_pos(ii,jj,:) = sum(kurtosis_pos) + sum(constraint3_pos);
            clear diffusino kurtosis constraint3
            
            %% computes the gradient of the physical relevance
            if (nargout >1)   
                grad_3_kurtosis_tmp = (-squeeze(bvecmatp(ii,jj,:,7:21))/0.01).*repmat(kurtosis_pos,[1 15]);
                grad_3_kurtosis(ii,jj,:) = col(sum(grad_3_kurtosis_tmp,1));
                
                grad_3_constraint3_tmp = (squeeze(bvecmatp(ii,jj,:,7:21))/0.01).*repmat(constraint3_pos,[1 15]);
                grad_3_constraint3(ii,jj,:) = col(sum(grad_3_constraint3_tmp,1));
 
                grad_3_diffusion_tmp = (-3*squeeze(bvecmatp(ii,jj,:,1:6))/max_bval/0.01).*repmat(constraint3_pos,[1 6]);
                grad_3_diffusion(ii,jj,:) = col(sum(grad_3_diffusion_tmp,1));        
                
                clear kurtosis_pos grad_3_kurtosis_tmp constraint3_pos grad_3_constraint3_tmp grad_3_diffusion_tmp
            end
        end
    end
end
temp5 = -0.5*(s./sigma_map).^2;               
sig_NCEXP = temp4*real(ppval(pp_initial(log2(Ncoils)+1),-temp5)).*sigma_map;clear temp5  
c1=sum(-2*im_r_w.*sig_NCEXP,3) + sum(sig_NCEXP.^2,3) + im_r_w_square;
c1(isnan(c1))=0;c1(isinf(c1))=0;
f1 = sum(c1(:));
tmp_g = s;
clear s sig_NCEXP

s_pos = sum(s_pos,3).*mask;
f3 = 0.0000001*sum(s_pos(:));
clear s_pos

%% TV term
tensorNorm(1) = norm(col(xp(:,:,1)));
tensorNorm(2) = norm(col(xp(:,:,2:7)))/sqrt(6);
tensorNorm(3) = norm(col(xp(:,:,8:22)))/sqrt(15);
scaling(1) = 1;
scaling(2:7) = tensorNorm(1)/tensorNorm(2);
scaling(8:22) = tensorNorm(1)/tensorNorm(3);

for kk = 1:22
    lambda1(kk) = lambda*scaling(kk);
end
clear scaling

f2 = 0;
for kk = 1:22
    Dxp(:,:,:,kk) = TVOP*squeeze(xp(:,:,kk));
    Dxp_tmp = squeeze(Dxp(:,:,:,kk)).*repmat(mask,[1 1 2]);
    f2 = f2 + lambda1(kk)*sum((conj(Dxp_tmp(:)).*Dxp_tmp(:)+1e-8).^(1/2));    
    clear Dxp_tmp
end

%% total objective function
feval = f1 + f2 + f3;



%% gradient
if ( nargout >1 )
    grad_3 = zeros(m,n,22);
for kk = 1:22  
    %%%%computes the gradient of the data consistency
    s = exp(bmatp(:,:,:,kk)*1e-6).*tmp_g;
    temp5 = -0.5*(s./sigma_map).^2;clear s
    sig_NCEXP = temp4*real(ppval(pp_initial(log2(Ncoils)+1),-temp5)).*sigma_map;   clear temp5          
    c1_g=sum(-2*im_r_w.*sig_NCEXP,3) + sum(sig_NCEXP.^2,3) + im_r_w_square;clear sig_NCEXP
    c1_g(isnan(c1_g))=0;c1_g(isinf(c1_g))=0;
    grad_1(:,:,kk) = (c1_g - c1)/1e-6;clear c1_g 
        
    %%%%%compute gradient of TV operator
    Dxp_tmp = squeeze(Dxp(:,:,:,kk));
    grad_2(:,:,kk) = TVOP'*(real(Dxp_tmp).*(Dxp_tmp.*conj(Dxp_tmp)+1e-8).^(-0.5)).*mask*lambda1(kk);clear Dxp_tmp
    
end
clear bmatp tmp_g im_r_pad_N Dxp c1 tmp_g_s_pos s_pos

%%%%% computes the gradient of the physical relevance
grad_3(:,:,8:22) = 0.0000001*(grad_3_kurtosis + grad_3_constraint3);
grad_3(:,:,2:7) = 0.0000001*(grad_3_diffusion);
clear grad_3_diffusion grad_3_constraint3 grad_3_kurtosis

%% total gradient   
grad = grad_1 + grad_2 + grad_3 ;
clear grad_1 grad_2 grad_3
                 
    
end

22;





