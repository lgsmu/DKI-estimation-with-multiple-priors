function sigma = cal_sigma_bacg(im_r,Ncoils,grad,param)

bval = grad(:,4);
ind = (bval>0.5)&(bval<3.5);
ind2 = find(ind==1);

tmp = squeeze(im_r(param.xstart:param.xstart+param.xlength,param.ystart:param.ystart+param.ylength,1,ind2));

sigma = sqrt(mean(tmp(:).^2)/2/Ncoils);

end