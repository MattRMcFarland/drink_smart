%% SVD analysis
close all; clear all;
X = importdata('data/nonnormalized_collected_reviews.csv',',');

sparcity = length(find(~isnan(X.data))) / (size(X.data,1) * size(X.data,2))

% ---- SET PARAMETERS HERE ---- %
K = 3;
params.max_iterations = 1200;
params.threshold = 1e-4;
params.step_size = 1e-3;
params.batch_size = 10;
w.k_U = 1;
w.k_V = 1;

% strip any reviewers who didn't have any beers reviewed in this set
to_remove = sum(~isnan(X.data),2) == 0;
X.data(to_remove,:) = [];
fprintf('%d reviewers were removed because they lacked any reviews in this set.\n',...
    sum(to_remove));

holdout_users = 10;
Xholdout = X.data(1:holdout_users,:);
X.data(end+holdout_users:end,:) = [];

% get test points
test_pcent = .2;
test_mask = ~isnan(X.data) .* (rand(size(X.data,1),size(X.data,2)) < test_pcent);

Xtrain = ~test_mask .* X.data;
Xtrain(test_mask == 1) = NaN;
Xtest = test_mask .* X.data;
Xtest(~test_mask == 1) = NaN;

% remove users who don't have both testing and training data
to_remove_cumulative = (sum(~isnan(Xtrain),2) == 0) | ...
                       (sum(~isnan(Xtest),2) == 0);
Xtrain(to_remove_cumulative,:) = [];
Xtest(to_remove_cumulative,:) = [];
test_mask(to_remove_cumulative,:) = [];
fprintf('%d more reviewers were removed b/c they did not have enough test and training reviews.\n',...
    sum(to_remove_cumulative));

% Now execute SVD
d = size(Xtrain,2);
n = size(Xtrain,1);

init_lim = 1;
%initialize the parameter to some small random value     
svd.U = unifrnd( -init_lim, init_lim, [n, K]); 
svd.V = unifrnd( -init_lim, init_lim, [d, K]);

% train without regularization
[mse, best_UV] = svd_train(Xtrain,svd,params);

% train and optimize with regularization
%[mse, best_UV] = svd_reg_train(Xtrain, svd, params, w);

figure();
plot(mse,'rx');
xlabel('iteration');
ylabel('MSE');
title_str = sprintf('n = %d, d = %d\nEnd MSE: %.4f -- %d iterations\n|U| = %.1f, |V| = %.1f',...
    n,d,mse(end),length(mse),norm(best_UV.U),norm(best_UV.V));
title(title_str);

print -dpng 'figures/regularized_svd'

%% now test U and V
[beer_avg_mse, user_avg_mse, beer_mse_var, user_mse_var] = get_avg_baseline(Xtest)
[test_error, user_err_avgs, user_error_var] = svd_testing_error(Xtest, best_UV.U, best_UV.V);

fprintf('Testing error is %.4f\n',test_error);

figure();
subplot(1,2,1)
hist(user_err_avgs,100);
title('average errors for users')

subplot(1,2,2)
hist(user_error_var,100);
title('variance of errors for users')

% --- predict for holdout data set and get error --- 
%holdout_error = svd_predict_and_test(Xtest,best_UV.V);
%fprintf('Holdout error is %.4f\n',holdout_error)




