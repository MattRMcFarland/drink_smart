function [Best_K, errors] = find_best_K(X, params, w, K_vec, test_pcent)
%% Cross validate testing for the right K, given a w vector
% Note: uses regularization of SVD
% INPUTS
%   Xtrain - n x d
%   params.batch_size - 1x1
%   params.stepsize - 1x1
%   params.max_iterations = 1x1
%   w.k.U - 1 x 1
%   w.k_V - 1 x 1
%   K_vec - 1 x m of K values to try

Nfold = 3;

errors = zeros(1,length(K_vec));
for i = 1:length(K_vec)
    
    % K implied by size of U and V
    K = K_vec(i);
    
    train_func = @(X, svd)svd_reg_train(X, svd, params, w);
    eval_func = @(X, svd)svd_testing_error(X,svd.U,svd.V);
    errors(i) = cross_validation(X, Nfold, train_func, eval_func, K);
end

[~, best_i] = min(errors);

Best_K = K_vec(best_i);
end
    