function [mse, beer_avgs, user_bias, test_residuals] = bias_baseline(Xtrain, Xtest)
%% guesses based on average movie rating and user bias
% Note: do not feed this function centered data
% INPUT:
%   Xtrain - n x d 
%   Xtest - n x d
%
% OUTPUT:
%   mse - 1 x 1

n = size(Xtrain,1);
d = size(Xtrain,2);

train_I = ~isnan(Xtrain);
Xtrain(isnan(Xtrain)) = 0;

test_I = ~isnan(Xtest);

% start by getting beer avgs and user bias off of average
% use crazy netflix formula: http://sifter.org/~simon/journal/20061211.html
GlobalMean = sum(sum(Xtrain,2)) / sum(sum(train_I,2));
global_variance = sum(sum((train_I .* (Xtrain - GlobalMean)).^2,2)) ./ (sum(sum(train_I,2)) - 1);
b_avgs_naive = sum(Xtrain,1) ./ sum(train_I,1);
beer_variance = sum((train_I .* (repmat(b_avgs_naive,n,1) - Xtrain)).^2,1) ./ (sum(train_I,1)-1);
K = beer_variance / global_variance;

% paper uses an alternative of K = 25 ...
%K = 25;

beer_avgs = (GlobalMean * K + sum(Xtrain,1)) ./ (K + sum(train_I,1));
user_bias = sum(train_I .* (Xtrain - repmat(beer_avgs,n,1)),2) ./ sum(train_I,2);

predictions = repmat(beer_avgs,n,1) + repmat(user_bias,1,d);
test_residuals = Xtest - predictions;

test_residuals_copy = test_residuals;
test_residuals_copy(isnan(test_residuals_copy)) = 0;
% calculate mse
mse = sum(sum((test_I .* test_residuals_copy).^2,2)) ./ sum(sum(test_I,2));
end
