function [beer_avg_mse, user_avg_mse, global_mse, beer_mse_var, user_mse_var, global_mse_var] = get_avg_baseline(X)
%% calculates baseline prediction RMSE (average rating for beer, average for user, average globally)
% INPUT:
%   X - n x d
%
% OUTPUT:
%   beer_avg_mse - 1 x 1
%   user_avg_mse - 1 x 1

n = size(X,1);
d = size(X,2);

I = ~isnan(X);
X(isnan(X)) = 0;

user_avgs = sum(I .* X,2) ./ sum(I,2);
beer_avgs = sum(I .* X,1) ./ sum(I,1);
global_avg = sum(sum(X,1)) / sum(sum(I,1));

users_avg = sum((I .* (X - repmat(user_avgs,1,d))).^2,2) ./ sum(I,2);
user_avg_mse = mean(users_avg);
user_mse_var = var(users_avg);
%user_mse_var = mean(sum((I .* (X - repmat(users_avg,1,d)).^2), 2) ./ (sum(I,2) - 1));

beers_avg = sum((I .* (X - repmat(beer_avgs,n,1))).^2,1) ./ sum(I,1);
beer_avg_mse = mean(beers_avg);
beer_mse_var = var(beers_avg);
%beer_mse_var = mean(sum((I .* (X - repmat(beers_avg,n,1)).^2),1) ./ (sum(I,1) - 1));

global_mse = sum(sum((I .* (X - ones(n,d) * global_avg)).^2,1)) ./ sum(sum(I,1));
global_mse_var = sum(sum((I .* (X - ones(n,d) * global_avg)).^2,1)) ./ (sum(sum(I,1)) -1);

