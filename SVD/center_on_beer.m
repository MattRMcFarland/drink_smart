function [centered_data] = center_on_beer(X)
%% centers each column to zero mean
% INPUT:
%   X - n x m
%
% OUTPUT:
%   centered_data - n x m

I = ~isnan(X);
X_zero = X;
X_zero(isnan(X_zero)) = 0;

means = sum(I .* X_zero,1) ./ sum(I,1);
centered_data = X - repmat(means,size(X,1),1);
end