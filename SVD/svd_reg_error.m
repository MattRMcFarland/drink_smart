function [mse, grad] = svd_reg_error(X, svd)
%% calculates error and gradient of SVD for given 
% INPUTS:
%   X - n x d (users by items)
%   svd.U - n x k 
%   svd.V - d x k
%   svd.k.U - 1 x 1
%   svd.k_V - 1 x 1
% OUTPUTS:
%   mse - 1 x 1 mean squared error + L2 regularization of U and V
%   grad.U - n x k 
%   grad.V - d x k

n = size(X, 1);
d = size(X, 2);
k = size(svd.V, 2);

I = ~isnan(X);      % get indicator
X(isnan(X)) = 0;    % replace NaNs
user_observation_count = sum(I,2);
predictions = svd.U * svd.V';

user_mse = sum( (I .* (X - predictions)).^2, 2) ./ user_observation_count;
mse = mean(user_mse) + svd.k_U * norm(svd.U) + svd.k_V * norm(svd.V);

grad.U = -2 * I .* (X - predictions) * svd.V + 2 * svd.k_U * svd.U;
grad.V = -2 * I' .* (X - predictions)' * svd.U + 2 * svd.k_V * svd.V;
