function [mse, grad] = svd_error_improved(X, svd, w)
%% calculates error and gradient of SVD for given 
% INPUTS:
%   X - n x d (users by items)
%   svd.U - n x k 
%   svd.V - d x k
%   svd.c - n x 1
%   svd.d - 1 x d
%
%   w.k_U - 1 x 1
%   w.k_V - 1 x 1
%   w.k_c - 1 x 1
%   w.k_d - 1 x 1
%
% OUTPUTS:
%   mse - 1 x 1 mean squared error + L2 regularization of U and V
%   grad.U - n x k 
%   grad.V - d x k
%   grad.c - 1 x n
%   grad.d - d x 1

if nargin < 3
    w.k_U = 0;
    w.k_V = 0;
    w.k_c = 0;
    w.k_d = 0;
end

n = size(X, 1);
d = size(X, 2);
k = size(svd.V, 2);

I = ~isnan(X);      % get indicator
X(isnan(X)) = 0;    % replace NaNs
UV_prod = svd.U * svd.V';
predictions = UV_prod + repmat(svd.c,1,d) + repmat(svd.d,n,1);

mse = sum(sum(I .* (X - predictions).^2,2)) ./ sum(sum(I,2)) + ...
    w.k_U * norm(svd.U) + w.k_V * norm(svd.V);

grad.U = -2 * I .* (X - predictions) * svd.V +  2 * w.k_U * svd.U;
grad.V = -2 * I' .* (X - predictions)' * svd.U + 2 * w.k_V * svd.V;
grad.c = -2 * (sum(I .* (X - (repmat(svd.d,n,1) - UV_prod)),2) - svd.c);
grad.d = -2 * (sum(I .* (X - (repmat(svd.c,1,d) - UV_prod)),1) - svd.d);


