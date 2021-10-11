function [im_r_w im_r_w_square ] = fun_objective_factor(nss_weight,im_r,t,Mask)              
 

[m n ndwi]=size(im_r);

 for i = 1:ndwi
   im_r_pad(:,:,i) = padarray(im_r(:,:,i),[t t],'symmetric');
 end
im_r_w=zeros([m n ndwi]);
im_r_w_square=zeros([m n ndwi]);
for i=1:m
    for j=1:n
        if Mask(i,j)>0
        tmp=nss_weight(i,j,:,:);
        tmp=reshape(tmp,[2*t+1 2*t+1]);
        tmp=repmat(tmp,[1 1 ndwi]);
        cc=tmp.*im_r_pad(i:i+2*t,j:j+2*t,:);
        im_r_w(i,j,:)=squeeze(sum(sum(cc,1),2));
        cc=tmp.*im_r_pad(i:i+2*t,j:j+2*t,:).^2;
        im_r_w_square(i,j,:)=squeeze(sum(sum(cc,1),2));
        end
    end
end
        