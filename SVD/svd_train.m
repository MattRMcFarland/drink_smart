function [mse, svd] = svd_train(X, svd_in, params)
%% Uses gradient descent to find optimal U and V for SVD
% INPUTS:
%   X - n x d
%   svd.U - n x k
%   svd.V - d x k
%   params.batch_size - 1x1
%   params.stepsize - 1x1
%   params.max_iterations = 1x1
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
    
    for i = 1:params.batch_size:size(X, 1)
    %for i = 1:4
        % define the batch set        
        batch = i:min(i+params.batch_size-1, size(X, 1));
        Xbatch = X(batch, :); 
        
        svd_batch.U = svd.U(batch,:);
        svd_batch.V = svd.V;
        
        % update U 
        [~, grad] = svd_error(Xbatch,svd_batch);
        svd.U(batch,:) = svd.U(batch,:) - (params.step_size / params.batch_size) * grad.U;
        svd.V = svd.V - (params.step_size / params.batch_size) * grad.V;
    end
%     [~,grad] = svd_error(X,svd);
%     svd.U = svd.U - (params.step_size) * grad.U;
%     svd.V = svd.V - (params.step_size) * grad.V;      

    % keep track of the MSE across iterations
    mse = [mse, svd_error(X,svd)];      
    fprintf('svd_train: iter %d, mse %.5f\n', iter, mse(end));
       
    % stopping criterion
    % Go until threshold is reached
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