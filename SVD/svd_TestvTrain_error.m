%% SVD analysis
close all; clear all;
X = importdata('data/nonnormalized_collected_reviews.csv',',');

% ---- SET PARAMETERS HERE ---- %
K = 50;
params.max_iterations = 1200;
params.threshold = 1e-3;
params.step_size = 1e-3;
params.batch_size = 10;

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

init_lim = .005;
%initialize the parameter to some small random value     
svd.U = unifrnd( -init_lim, init_lim, [n, K]); 
svd.V = unifrnd( -init_lim, init_lim, [d, K]);

% get testing vs. training errors
params.max_iterations = 25;

iters = 20;
training_error = [];
testing_error = zeros(iters,2);
for i = 1:iters
    % train
    [mse, UV] = svd_train_from_start_point(Xtrain, svd, params);
    training_error = [training_error, mse];
    svd = UV;
    
    % calculate testing errors
    testing_error(i,1) = [i * params.max_iterations];
    testing_error(i,2) = svd_testing_error(Xtest, UV.U, UV.V);    
end

figure(); hold on;
plot(mse,'rx');
plot(testing_error(:,1),testing_error(:,2),'bo')
legend('training error','testing error')
xlabel('iteration');
ylabel('MSE');
title_str = sprintf('n = %d, d = %d\nEnd MSE: %.4f -- %d iterations\n|U| = %.1f, |V| = %.1f',...
    n,d,mse(end),length(mse),norm(best_UV.U),norm(best_UV.V));
title(title_str);

print -dpng 'figures/train_and_test_mse_comparison'





