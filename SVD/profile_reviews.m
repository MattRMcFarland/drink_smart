function [global_avg, beer_avgs, user_avgs, beers_var, users_var] = profile_reviews(X)
%% characterizes review dataset
% INPUT:
%   X - n x d
%
% OUTPUT:
%   global_avg - 1 x 1
%   beer_avgs - 1 x d
%   user_avgs - n x 1
%   beer_var - 1 x d
%   user_var - n x 1

I = ~isnan(X);
X(isnan(X)) = 0;

global_avg = sum(sum(X)) / sum(sum(I));
beer_avgs = sum(X,1) ./ sum(I,1);
user_avgs = sum(X,2) ./ sum(I,2);

beers_var = sum((I.*(X - repmat(beer_avgs,size(X,1),1))).^2,1) ./ (sum(I,1)-1);
users_var = sum((I.*(X - repmat(user_avgs,1,size(X,2)))).^2,2) ./ (sum(I,2)-1);
end