function [mse, baseline_bias_mse] = calc_bias_error(Xtest, beer_avgs, user_bias, UV_prod)
%% Calculates the baseline and predicted error
% INPUT:
%   Xtest - n x d
%   beer_avgs - 1 x d
%   user_bias - n x 1
%   UV - n x d
%
% OUTPUT
%   mse - 1 x 1
%   baseline_bias_mse - 1 x 1

n = size(Xtest,1);
d = size(Xtest,2);
test_predictions_base = repmat(beer_avgs,n,1) + repmat(user_bias,1,d); 
test_predictions = test_predictions_base + UV_prod;
test_I = ~isnan(Xtest);
Xtest(isnan(Xtest)) = 0;
baseline_bias_mse = sum(sum((test_I .* (Xtest - test_predictions_base)).^2,2)) ./ sum(sum(test_I,2));
mse = sum(sum((test_I .* (Xtest - test_predictions)).^2,2)) ./ sum(sum(test_I,2));
end