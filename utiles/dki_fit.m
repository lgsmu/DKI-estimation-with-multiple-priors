function [b0, dt_21,dt_22] = dki_fit(dwi, grad, mask, constraints, outliers, maxbval)
    % Diffusion Kurtosis Imaging tensor estimation using 
    % (constrained) weighted linear least squares estimation 
    % -----------------------------------------------------------------------------------
    % please cite:  Veraart, J.; Sijbers, J.; Sunaert, S.; Leemans, A. & Jeurissen, B.,
    %               Weighted linear least squares estimation of diffusion MRI parameters: 
    %               strengths, limitations, and pitfalls. NeuroImage, 2013, 81, 335-346 
    %------------------------------------------------------------------------------------
    % 
    % Usage:
    % ------
    % [b0, dt] = dki_fit(dwi, grad [, mask [, constraints]])
    %
    % Required input: 
    % ---------------
    %     1. dwi: diffusion-weighted images.
    %           [x, y, z, ndwis]
    %       
    %       Important: We recommend that you apply denoising, gibbs correction, motion-
    %       and eddy current correction to the diffusion-weighted image
    %       prior to tensor fitting. Thes steps are not includede in this
    %       tools, but we are happy to assist (Jelle.Veraart@nyumc.org).
    %
    %     2. grad: diffusion encoding information (gradient direction 'g = [gx, gy, gz]' and b-values 'b')
    %           [ndwis, 4] 
    %           format: [gx, gy, gx, b]
    %
    % Optional input:
    % ---------------
    %    3. mask (boolean; [x, y, x]), providing a mask limits the
    %       calculation to a user-defined region-of-interest.
    %       default: mask = full FOV
    %
    %    4 . constraints (boolean; [1, 3] as in [c1, c2, c3]), imposes
    %       user-defined constraint to the weighted linear leasts squares
    %       estimation of the diffusion kurtosis tensor.
    %       Following constraints are available:
    %           c1: Dapp > 0
    %           c2: Kapp > 0
    %           c3: Kapp < b/(3*Dapp)
    %       default: [0 1 0]
    %     5. maxbval (scalar; default = 2.5ms/um^2), puts an upper bound on the b-values being
    %     used in the analysis.
    %
    % Copyright (c) 2017 New York University and University of Antwerp
    %
    % This Source Code Form is subject to the terms of the Mozilla Public
    % License, v. 2.0. If a copy of the MPL was not distributed with this file,
    % You can obtain one at http://mozilla.org/MPL/2.0/
    % 
    % This code is distributed  WITHOUT ANY WARRANTY; without even the 
    % implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
    % 
    % For more details, contact: Jelle.Veraart@nyumc.org 

    
    %% limit DKI fit to b=3000
    bval = grad(:, 4);
    order = floor(log(abs(max(bval)+1))./log(10));
    if order >= 2
        grad(:, 4) = grad(:, 4)/1000;
        bval = grad(:, 4);
    end
    
    if ~exist('maxbval','var') || isempty(maxbval)
        maxbval = 3;
    end
    list = bval<=maxbval;
    dwi = dwi(:,:,:,list);
    grad = grad(list, :);
             
    %% Vectorization of dwi
    dwi = double(dwi);
    dwi(dwi<=0)=eps;
    v = false;
    if ~ismatrix(dwi)
        v = true;
        if ~exist('mask','var') || isempty(mask)
            mask = ~isnan(dwi(:,:,:,1));
        end
        dwi_ = zeros([size(dwi,4) sum(mask(:))], class(dwi));
        for k = 1:size(dwi,4);
            tmp = dwi(:,:,:,k);
            dwi_(k,:) = tmp(mask(:));
        end
        dwi = dwi_;
        clear dwi_ tmp
    end
    [ndwi, nvox] = size(dwi);
      
   

%% scaling
    scaling = false;
    if numel(dwi(dwi<1))/numel(dwi) < 0.001
        dwi(dwi<1)=1;
    else
        scaling = true;
        if max(bval)<10
            tmp = dwi(bval<0.05,:);
        else
            tmp = dwi(bval<50,:);
        end
        sc = median(tmp(:));
        dwi(dwi<sc/1000) = sc/1000;
        dwi = dwi*1000/sc;
    end    
         
    %% parameter checks 
    grad = double(grad);
    normgrad = sqrt(sum(grad(:, 1:3).^2, 2)); normgrad(normgrad == 0) = 1;
    grad(:, 1:3) = grad(:, 1:3)./repmat(normgrad, [1 3]);
    grad(isnan(grad)) = 0;


    if exist('constraints', 'var') && ~isempty(constraints) && numel(constraints)==3
  
    else
        constraints = [0 1 0];
    end
    constraints = constraints > 0;
    %% tensor fit
    [D_ind, D_cnt] = createTensorOrder(2);
    [W_ind, W_cnt] = createTensorOrder(4);
    
    bS = ones(ndwi, 1);
    bD = D_cnt(ones(ndwi, 1), :).*grad(:,D_ind(:, 1)).*grad(:,D_ind(:, 2));
    bW = W_cnt(ones(ndwi, 1), :).*grad(:,W_ind(:, 1)).*grad(:,W_ind(:, 2)).*grad(:,W_ind(:, 3)).*grad(:,W_ind(:, 4));
    
    b = [bS, -bval(:, ones(1, 6)).*bD, (bval(:, ones(1, 15)).^2/6).*bW];
   

    % unconstrained LLS fit
    dt = b\log(dwi);
    w = exp(b*dt);

    nvoxels = size(dwi,2);
     
    outliers = false(size(dwi));

    
    % WLLS fit initialized with LLS   
    if any(constraints) 
       
        dir(1:ndwi,1:3) = grad(:, 1:3); 

        ndir = size(dir, 1);
        C = [];
        if constraints(1)>0
            C = [C; [zeros(ndir, 1), D_cnt(ones(ndir, 1), :).*dir(:,D_ind(:, 1)).*dir(:,D_ind(:, 2)), zeros(ndir, 15)]];
        end
        if constraints(2)>0
            C = [C; [zeros(ndir, 7), W_cnt(ones(ndir, 1), :).*dir(:,W_ind(:, 1)).*dir(:,W_ind(:, 2)).*dir(:,W_ind(:, 3)).*dir(:,W_ind(:, 4))]];
        end
        if constraints(3)>0
%             C = [C; [zeros(ndir, 1), 3/max(bval)*D_cnt(ones(ndir, 1), :).*dir(:,D_ind(:, 1)).*dir(:,D_ind(:, 2)), -W_cnt(ones(ndir, 1), :).*dir(:,W_ind(:, 1)).*dir(:,W_ind(:, 2)).*dir(:,W_ind(:, 3)).*dir(:,W_ind(:, 4))]];
            C = [C; [zeros(ndir, 1), 3/2*D_cnt(ones(ndir, 1), :).*dir(:,D_ind(:, 1)).*dir(:,D_ind(:, 2)), -W_cnt(ones(ndir, 1), :).*dir(:,W_ind(:, 1)).*dir(:,W_ind(:, 2)).*dir(:,W_ind(:, 3)).*dir(:,W_ind(:, 4))]];

        end
        d = zeros([1, size(C, 1)]);
        options = optimset('Display', 'off', 'Algorithm', 'interior-point', 'MaxIter', 22000, 'TolCon', 1e-12, 'TolFun', 1e-12, 'TolX', 1e-12, 'MaxFunEvals', 220000);
        for i = 1:nvoxels
            try
                in_ = outliers(:, i) == 0;
                wi = w(:,i); Wi = diag(wi(in_));             
                dt(:, i) = lsqlin(Wi*b(in_, :),Wi*log(dwi(in_,i)),-C, d, [],[],[],[],[],options);
            catch
                dt(:, i) = 0;
            end
        end
    else
        for i = 1:nvoxels
            in_ = outliers(:, i) == 0;
            b_ = b(in_, :);
            if isempty(b_) || cond(b(in_, :))>1e15
                dt(:, i) = NaN
            else
                wi = w(:,i); Wi = diag(wi(in_)); 
                logdwii = log(dwi(in_,i));
                dt(:,i) = (Wi*b_)\(Wi*logdwii);
            end
        end
    end

    %% unscaling
    if scaling
        dt(1,:) = dt(1,:)+log(sc/1000);
    end
    
    dt_22 = dt;    
    b0 = exp(dt(1,:));
    dt_21 = dt(2:22, :);
    
    D_apprSq = 1./(sum(dt_21([1 4 6],:),1)/3).^2;
    dt_21(7:21,:) = dt_21(7:21,:) .* D_apprSq(ones(15,1),:);
    
    
%% Unvectorizing output variables
    if v
       dims = size(mask);
       
       % unvec dt_21
       dt_21_ = zeros([dims], class(dt_21));
       for k = 1:size(dt_21,1)
           tmp = zeros(dims, class(dt_21));
           tmp(mask) = dt_21(k,:);
           dt_21_(:,:,k) = tmp;
       end
       dt_21 = dt_21_; clear dt_21_;
    
       
       % unvec dt_22
       dt_22_ = zeros([dims], class(dt_22));
       for k = 1:size(dt_22,1)
           tmp = zeros(dims, class(dt_22));
           tmp(mask) = dt_22(k,:);
           dt_22_(:,:,k) = tmp;
       end
       dt_22 = dt_22_; clear dt_22_;
       
       
       % unvec b0
       b0_ = zeros([dims 1], 'double'); 
       b0_(mask) = b0; 
       b0 = b0_; 
       clear b0_;

    end
    
    
    
end






