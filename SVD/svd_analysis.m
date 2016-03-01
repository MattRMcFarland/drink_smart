%% SVD analysis
close all; clear all;

X = importdata('small_collected_reviews.csv',',');

% strip any reviewers who didn't have any beers reviewed in this set
to_remove = sum(~isnan(X.data),2) == 0;
X.data(to_remove,:) = []

K = 100;
d = size(X.data,2);
n = size(X.data,1);

% [U,S,V] = svds(X.data,K);

init_lim = 1;
%initialize the parameter to some small random value     
svd.U = unifrnd( -init_lim, init_lim, [n, K]); 
svd.V = unifrnd( -init_lim, init_lim, [d, K]);

% set parameters
params.max_iterations = 50;
params.threshold = 1e-3;
params.step_size = 1e-8;
params.batch_size = 10;

[mse, best_UV] = svd_train(X.data, svd, params);






