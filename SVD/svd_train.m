function [mse, svd] = svd_train(X, grad_0, params)
%% Uses gradient descent to find optimal U and V for SVD
% INPUTS:
%   X - n x d
%   grad_0.U - n x k
%   grad_0.V - d x k
%   params.batch_size - 1x1
%   params.stepsize - 1x1
%   params.max_iterations = 1x1
%
% OUTPUTS:
%   mse - row vector with evolved mse errors
%   grad.U - n x k  [optimized U]
%   grad.V - d x k  [optimized V]

k = size(grad_0.V,2);
d = size(X,2);
n = size(X,1);
% [U,S,V] = svds(X.data,K);

[~, svd] = svd_error(X, grad_0);
mse = [];
for iter = 1:params.max_iterations  
    
    for i = 1:params.batch_size:size(X, 1)
        % define the batch set        
        batch = i:min(i+params.batch_size-1, size(X, 1));
        Xbatch = X(batch, :); 
        
        svd_batch.U = svd.U(batch,:);
        svd_batch.V = svd.V;
        
        % update U and V
        [~, grad] = svd_error(Xbatch,svd_batch);
        svd.U(batch,:) = svd.U(batch,:) - (params.step_size / params.batch_size) * grad.U;        
    end
    svd.V = svd.V - (params.step_size) * grad.V;      
    
    % decrease the step size -- remove
    % params.step_size = params.step_size / iter  
%     if iter == 200
%         params.step_size = params.step_size * 10;
%         fprintf('increasing step size to %f\n',params.step_size)
%     end
    
    % keep track of the MSE across iterations
    mse = [mse, svd_error(X,svd)];      
    fprintf('svd_train: iter %d, mse %.4f\n', iter, mse(end));
       
    % threshold = 1e-3;
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