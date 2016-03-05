function [centered_data] = center_globally(X)
%% centers each column to zero mean
% INPUT:
%   X - n x m
%
% OUTPUT:
%   centered_data - n x m

I = ~isnan(X);
X_zero = X;
X_zero(isnan(X_zero)) = 0;

global_mean = sum(sum(I .* X_zero,1)) / sum(sum(I,1));
centered_data = X - ones(size(X,1),size(X,2)) * global_mean;
end