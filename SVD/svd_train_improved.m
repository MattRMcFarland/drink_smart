function [mse, svd] = svd_train_improved(X, svd_in, params)
%% Uses gradient descent to find optimal U and V for SVD
% INPUTS:
%   X - n x d
%   svd.U - n x k
%   svd.V - d x k
%   svd.c - n x 1
%   svd.d - 1 x d
%   svd.global_mean - 1 x 1
%   params.batch_size - 1 x 1
%   params.stepsize - 1 x 1
%   params.max_iterations = 1 x 1
%
% OUTPUTS:
%   mse - row vector with evolved mse errors
%   grad.U - n x k  [optimized U]
%   grad.V - d x k  [optimized V]

k = size(svd_in.V,2);
d = size(X,2);
n = size(X,1);

svd = svd_in;
mse = [];
for iter = 1:params.max_iterations  
    
%     for i = 1:params.batch_size:size(X, 1)
%         % define the batch set        
%         batch = i:min(i+params.batch_size-1, size(X, 1));
%         Xbatch = X(batch, :); 
%         
%         svd_batch.U = svd.U(batch,:);
%         svd_batch.V = svd.V;
%         
%         % update U 
%         [~, grad] = svd_error(Xbatch,svd_batch);
%         svd.U(batch,:) = svd.U(batch,:) - (params.step_size / params.batch_size) * grad.U;        
%     end
    [~,grad] = svd_error_improved(X,svd);   % don't use regularized weights right now
    svd.U = svd.U - (params.step_size) * grad.U;
    svd.V = svd.V - (params.step_size) * grad.V;      
    svd.c = svd.c - (params.bias_step_size) * grad.c;
    svd.d = svd.d - (params.bias_step_size) * grad.d;
    
    % keep track of the MSE across iterations
    mse = [mse, svd_error(X,svd)];      
    fprintf('svd_train: iter %d, mse %.5f\n', iter, mse(end));
       
    % stopping criterion
    if (length(mse) > 1 && max(abs(mse(end) - mse(end-1))) < params.threshold) || ...
            (length(mse) > 1 && (mse(end) - mse(end-1) > 0))
        fprintf('svd_train: reached stopping threshold at iteration %d for with MSE %.4G\n',...
            iter,mse(end))
        break;
    end        
      
end
fprintf('stopping now.\nIterations: %d\nMagnitude of svd.U: %f\nMagnitude of svd.V: %d\n',...
    iter, norm(svd.U), norm(svd.V));

return;
end