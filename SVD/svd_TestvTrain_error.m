%% SVD analysis
close all; clear all;
% ---- SET PARAMETERS HERE ---- %
    % File and saved image?
    file_str = 'data/7k_users_800_beers.csv';
    % graph_str = 'figures/7k_users_800_beers_centered';

    % training parameters
    K = 10;
    params.max_iterations = 1200;
    params.threshold = 1e-5;
    params.step_size = 1e-3;
    params.batch_size = 10;

    % U and V regularization rates
    w.k_U = .02;
    w.k_V = .02;
    w.k_c = .02;
    w.k_d = .02;

    % test v. train split
    test_pcent = .20;

    % apply regularization? 1 if yes, 0 if no
    regularization_param = 1;
    
    % centering? 
    % 0 -> no 
    % 1 -> on users
    % 2 -> on beers (best?)
    % 3 -> center on beers and then users
    % 4 -> center on users and then beers
    % other -> globally    
    center_param = 2;

% ---- END PARAMETERS ---- %

% import and get raw stats on data
X = importdata(file_str,',');
[global_avg, beer_avgs, user_avgs, beers_var, users_var] = profile_reviews(X.data);

% report sparcity
sparcity = length(find(~isnan(X.data))) ./ (size(X.data,1)*size(X.data,2))

% strip any reviewers who didn't have any beers reviewed in this set
to_remove = sum(~isnan(X.data),2) == 0;
X.data(to_remove,:) = [];
fprintf('%d reviewers were removed because they lacked any reviews in this set.\n',...
    sum(to_remove));

% grab some users for holdout testing
holdout_users = 10;
Xholdout = X.data(1:holdout_users,:);
X.data(end+holdout_users:end,:) = [];

% get test points
test_mask = ~isnan(X.data) .* (rand(size(X.data,1),size(X.data,2)) < test_pcent);
Xtrain = ~test_mask .* X.data;
Xtrain(test_mask == 1) = NaN;           % mark testing spots as untried
Xtest = test_mask .* X.data;
Xtest(~test_mask == 1) = NaN;           % mark training spots as untried

% remove users who don't have both testing and training data
to_remove_cumulative = (sum(~isnan(Xtrain),2) == 0) | ...
                       (sum(~isnan(Xtest),2) == 0);
Xtrain(to_remove_cumulative,:) = [];
Xtest(to_remove_cumulative,:) = [];
test_mask(to_remove_cumulative,:) = [];
fprintf('%d more reviewers were removed b/c they did not have enough test and training reviews.\n',...
    sum(to_remove_cumulative));

% calculate  
[beer_avg_mse, user_avg_mse, global_mse, ...
    beer_mse_var, user_mse_var, global_mse_var] = get_avg_baseline(Xtest);

% center data
if center_param == 0                    % do nothing
    fprintf('not centering the data');  
elseif center_param == 1                % on user
    fprintf('centering on users\n');
    Xtrain = center_on_user(Xtrain);
    Xtest = center_on_user(Xtest);
elseif center_param == 2                % on beers
    fprintf('centering on beers\n');
    Xtrain = center_on_beer(Xtrain);
    Xtest = center_on_beer(Xtest);    
elseif center_param == 3                % on beers and then users
    fprintf('centering on beers\n');    
    Xtrain = center_on_beer(Xtrain);
    Xtest = center_on_beer(Xtest);
    fprintf('centering on users\n');   
    Xtrain = center_on_user(Xtrain);
    Xtest = center_on_user(Xtest);
elseif center_param == 4                % on users and then beers
    fprintf('centering on users\n');    
    Xtrain = center_on_user(Xtrain);
    Xtest = center_on_user(Xtest);
    fprintf('centering on beers\n');    
    Xtrain = center_on_beer(Xtrain);
    Xtest = center_on_beer(Xtest); 
else                                    % or globally
    fprintf('centering globally\n');
    Xtrain = center_globally(Xtrain);
    Xtest = center_globally(Xtest);
end

% Now execute SVD
d = size(Xtrain,2);
n = size(Xtrain,1);

%% ---- NEW INITIALIZATION STRATEGY ---- %%
% initialize each user's (U_i) preferences to 1
% initialize V_j to that beer's average rating

% [t_global_avg, t_beer_avgs, t_user_avgs, t_beers_var, t_users_var] ...
%     = profile_reviews(Xtrain);
% 
% noise = .01;
% svd.U = ones(n,K) + unifrnd( -noise, noise, [n, K]);
% svd.V = repmat(exp(-(1:K)),d,1) .* repmat(t_beer_avgs',1,K) + unifrnd( -noise, noise, [d, K]);     

init_lim = 1;
%initialize the parameter to some small random value     
svd.U = unifrnd( -init_lim, init_lim, [n, K]); 
svd.V = unifrnd( -init_lim, init_lim, [d, K]);

% get testing vs. training errors
params.max_iterations = 10;

iters = 50;
training_error = [];
testing_error = zeros(iters,2);
total_iterations = 0;
for i = 1:iters
    
    if (regularization_param == 0)
        [mse, UV] = svd_train(Xtrain,svd,params);
    else
        [mse, UV] = svd_reg_train(Xtrain,svd,params,w);
    end
    total_iterations = total_iterations + length(mse);
    training_error = [training_error, mse];
    svd = UV;
    
    % calculate testing errors
    testing_error(i,1) = total_iterations;
    testing_error(i,2) = svd_testing_error(Xtest, UV.U, UV.V);
    
    if (length(mse) == 2)
        break;
    end
end

% plot testing and trainin error
figure(); hold on;
plot(training_error,'r-');
plot(testing_error(1:i,1),testing_error(1:i,2),'bo')
xlabel('iteration');
ylabel('MSE');

legend('Training Error','Testing Error');
title('Testing and Training Error');

desc_str = sprintf(['n = %d, d = %d\nk = %d, sparcity = %.3f\n', ...
    'centering = %d; reg = %d,\n%d iterations\n|U| = %.1f, |V| = %.1f\n', ...
    'Training Error: %.4f\nTesting Error = %.4f\n'],...
    n, d, K, sparcity, center_param, regularization_param,...
    length(training_error), norm(UV.U), norm(UV.V), mse(end), testing_error(i,2));
ylim=get(gca,'ylim');
xlim=get(gca,'xlim');
text(.35*(xlim(2)-xlim(1))+xlim(1),.6*(ylim(2)-ylim(1))+ylim(1),desc_str);

% ---- CHANGE PRINTED MSE GRAPH NAME HERE ---- %
print -dpng 'figures/7k_users_800_beers_centered'

% Report baselines (averages
[test_error, user_err_avgs, user_error_var] = svd_testing_error(Xtest, UV.U, UV.V);

% beer_avg_mse, user_avg_mse, global_mse, ...
%     beer_mse_var, user_mse_var, global_mse_var
fprintf('Testing error is %.4f. Beer avg MSE: %.4f, User avg MSE: %.4f, Global avg MSE: %.4f\n',...
    test_error, beer_avg_mse, user_avg_mse, global_mse);

figure();
subplot(1,2,1)
hist(user_err_avgs,100);
title('average errors for users')

subplot(1,2,2)
hist(user_error_var,100);
title('variance of errors for users')




