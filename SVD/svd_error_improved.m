function [mse, grad] = svd_error_improved(X, svd, w)
%% calculates error and gradient of SVD for given 
% INPUTS:
%   X - n x d (users by items)
%   svd.U - n x k 
%   svd.V - d x k
%   svd.c - n x 1
%   svd.d - 1 x d
%   w.k_U - 1 x 1
%   w.k_V - 1 x 1
%   w.k_c - 1 x 1
%   w.k_d - 1 x 1
% OUTPUTS:
%   mse - 1 x 1 mean squared error + L2 regularization of U and V
%   grad.U - n x k 
%   grad.V - d x k
%   grad.c - 1 x n
%   grad.d - 1 x d

if nargin < 3
    w.k_U = 1;
    w.k_V = 1;
    w.k_c = 1;
    w.k_d = 1;
end

n = size(X, 1);
d = size(X, 2);
k = size(svd.V, 2);

I = ~isnan(X);      % get indicator
X(isnan(X)) = 0;    % replace NaNs
user_observation_count = sum(I,2);
predictions = svd.U * svd.V' + repmat(svd.c,1,d) + repmat(svd.d,n,1);

user_mse = sum( (I .* (X - predictions)).^2, 2) ./ user_observation_count;
mse = mean(user_mse);

grad.U = -2 * I .* (X - predictions) * svd.V;
grad.V = -2 * I' .* (X - predictions)' * svd.U;
grad.c = -2 * I' .* (X - predictions) - w.k_c * (
