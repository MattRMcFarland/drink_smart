function [mse, grad] = svd_error(X, svd)
%% calculates error and gradient of SVD for given 
% INPUTS:
%   X - n x d (users by items)
%   svd.U - n x k 
%   svd.V - d x k
% OUTPUTS:
%   mse - 1 x 1 mean squared error + L2 regularization of U and V
%   grad.U - n x k 
%   grad.V - d x k

n = size(X, 1);
d = size(X, 2);
k = size(svd.V, 2);

I = ~isnan(X);      % get indicator
X(isnan(X)) = 0;    % replace NaNs
predictions = svd.U * svd.V';

mse = sum(sum((I .* (X - predictions)).^2,2)) / sum(sum(I,2));

grad.U = -2 * (I .* (X - predictions)) * svd.V;
grad.V = -2 * (I .* (X - predictions))' * svd.U;

% grad.U = zeros(n,k);
% grad.V = zeros(d,k);
% 
% for i = 1:n
%     for j = 1:d
%         grad.U(i,:) = grad.U(i,:) + I(i,j) * ((X(i,j) - svd.U(i,:) * svd.V(j,:)') * svd.V(j,:));
%         grad.V(j,:) = grad.V(j,:) + I(i,j) * ((X(i,j) - svd.U(i,:) * svd.V(j,:)') * svd.U(i,:));
%     end
% end
% grad.U = -2 * grad.U;
% grad.V = -2 * grad.V;

