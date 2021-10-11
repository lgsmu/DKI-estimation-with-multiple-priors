function output = fun_weight_nss(input,t,f,h)
 
 % Size of the image
 [m n k]=size(input);
 
 % Memory for the output
 output=zeros(m,n,2*t+1,2*t+1);

 % Replicate the boundaries of the input image
 for i = 1:k
    input2(:,:,i) = padarray(input(:,:,i),[t+f t+f],'symmetric');
 end
 
 % Used kernel
 kernel = make_kernel(f);
 kernel3D = repmat(kernel,[1,1,k]);
 kernel3D = kernel3D / sum(sum(sum(kernel3D)));
 
 h = h.^2;
  
 for i=1:m
 for j=1:n
                 
         i1 = i+t+f;
         j1 = j+t+f;
                
         W1= input2(i1-f:i1+f, j1-f:j1+f, :);        
         wmax=0;   
         sweight=0;  
        
         for r=i1-t:1:i1+t
         for s=j1-t:1:j1+t                                              
                if(r==i1 && s==j1) continue; end;                            
                W2= input2(r-f:r+f,s-f:s+f, :);                                 
                d = sum(sum(sum(kernel3D.*(W1-W2).*(W1-W2))));
                
                w = exp(-d/h(i,j)) ;
                          
                if w>wmax                
                    wmax=w;                   
                end  
                output(i,j,r-i1+t+1,s-j1+t+1)=w;
         end 
         end
         output(i,j,t+1,t+1)=wmax;
         
                
         if output(i,j,t+1,t+1) ==0
             output(i,j,t+1,t+1) =1;
         end
         
         
         sweight_tmp=output(i,j,:,:);sweight=sum(sweight_tmp(:));
         output(i,j,:,:)=output(i,j,:,:)/sweight;
                         
 end
 end
 
function [kernel] = make_kernel(f)              
 
kernel=zeros(2*f+1,2*f+1);
if f>0
    for d=1:f    
      value= 1 / (2*d+1)^2 ;    
      for i=-d:d
      for j=-d:d
        kernel(f+1-i,f+1-j)= kernel(f+1-i,f+1-j) + value ;
      end
      end
    end    
    kernel = kernel ./ f;
else 
    kernel = 1;
end