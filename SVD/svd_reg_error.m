function [mse, grad] = svd_reg_error(X, svd, w)
%% calculates error and gradient of SVD for given 
% INPUTS:
%   X - n x d (users by items)
%   svd.U - n x k 
%   svd.V - d x k
%   w.k.U - 1 x 1
%   w.k_V - 1 x 1
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

mse = sum(sum((I .* (X - predictions)).^2, 2)) ./ sum(sum(I,1)) + ...
    w.k_U * norm(svd.U) + w.k_V * norm(svd.V);
 
%grad.U = -2 * I .* (X - predictions) * svd.V + 2 * svd.U;
%grad.V = -2 * I' .* (X - predictions)' * svd.U + 2 * svd.V;
       
grad.U = zeros(n,k);
grad.V = zeros(d,k);

for i = 1:n
    %grad.U(i,:) = grad.U(i,:) + -2 * I(i,:) .* (sum(X(i,:)' .* svd.V,1) - svd.U(i,:) * V' * V) + 2 * svd.U(i,:);
    grad.U(i,:) = -2 * sum(I(i,:) .* (X(i,:) - svd.U(i,:) * svd.V')' .* svd.V, 1) + 2 * svd.U(i,:);
end

for j = 1:d
    %grad.V(j,:) = grad.V(j,:) + -2 * I(:,j) .* (sum(X(:,j) .* svd.U,1) - 
    grad.V(j,:) = -2 * sum(I(:,j) .* (X(:,j) - svd.U * svd.V(j,:)') .* svd.U, 1) + 2 * svd.V(j,:);
end

% for i = 1:n
%     for j = 1:d
%         grad.U(i,:) = grad.U(i,:) + I(i,j) * ((X(i,j) - svd.U(i,:) * svd.V(j,:)') * svd.V(j,:));
%         grad.V(j,:) = grad.V(j,:) + I(i,j) * ((X(i,j) - svd.U(i,:) * svd.V(j,:)') * svd.U(i,:));
%     end
% end
% grad.U = -2 * grad.U + 2 * svd.U;
% grad.V = -2 * grad.V + 2 * svd.V;
