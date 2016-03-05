function [centered_data] = center_on_user(X)
%% centers each row to zero mean
% INPUT:
%   X - n x m
%
% OUTPUT:
%   centered_data - n x m

I = ~isnan(X);
X_zero = X;
X_zero(isnan(X_zero)) = 0;

means = sum(I .* X_zero,2) ./ sum(I,2);
centered_data = X - repmat(means,1,size(X,2));
end