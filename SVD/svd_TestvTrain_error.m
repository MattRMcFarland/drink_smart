%% SVD analysis
close all; clear all;
% ---- SET PARAMETERS HERE ---- %
    % File and saved image?
    % 7k_users_800_beers.csv
    % nonnormalized_collected_reviews.csv
    file_str = 'data/7k_nonnormalized_collected_reviews.csv';
    % graph_str = 'figures/7k_users_800_beers_centered';

    % test v. train split
    test_pcent = .10;    
    
    % take m least common beers (if -1, take whole set)
    m = -1;
    
    % holdout_users will be used for new user prediction
    holdout_users = 10;
    
    % training parameters
    K = 20;
    params.max_iterations = 1200;
    params.threshold = 1e-3;
    params.step_size = 1e-3;
    params.bias_step_size = 1e-5;
    params.batch_size = 10;

    % U and V regularization rates
    w.k_U = .02;
    w.k_V = .02;
    
    % type of training? Improved (and regularized svd) = 1
    svd_mode = 0;
    
    % apply regularization? 1 if yes, 0 if no
    regularization_param = 1;
    
    % centering? 
    % 0 -> none 
    % 1 -> on users
    % 2 -> on beers (best?)
    % 3 -> center on beers and then users
    % 4 -> center on users and then beers
    % 5 -> center with beer averages and then user biases
    % other -> globally    
    center_param = 5;

% ---- END PARAMETERS ---- %

%% ---- PREPROCESSING ---- %%
% import and get raw stats on data
X = importdata(file_str,',');
[global_avg, beer_avgs, user_avgs, beers_var, users_var] = profile_reviews(X.data);

% select only lowest rated beers if m > 0
I = ~isnan(X.data);
beer_observations = sum(I,1);
[common_beers, common_I] = sort(beer_observations,'ascend');
if m > 0
    X.data = X.data(:,common_I(1:m));
end

% report sparcity
sparcity = length(find(~isnan(X.data))) ./ (size(X.data,1)*size(X.data,2))

% strip any reviewers who didn't have any beers reviewed in this set
to_remove = sum(~isnan(X.data),2) == 0;
X.data(to_remove,:) = [];
fprintf('%d reviewers were removed because they lacked any reviews in this set.\n',...
    sum(to_remove));

% grab some users for holdout testing
holdout_i = randperm(size(X.data,1),holdout_users);
Xholdout = X.data(holdout_i,:);
X.data(holdout_i,:) = [];

% get test points
test_mask = ~isnan(X.data) .* (rand(size(X.data,1),size(X.data,2)) < test_pcent);
Xtrain = ~test_mask .* X.data;
Xtrain(test_mask == 1) = NaN;           % mark testing spots as untried
Xtest = test_mask .* X.data;
Xtest(~test_mask == 1) = NaN;           % mark training spots as untried

% remove users who don't have both testing and training data
to_remove_users = (sum(~isnan(Xtrain),2) == 0) | ...
                       (sum(~isnan(Xtest),2) == 0);
Xtrain(to_remove_users,:) = [];
Xtest(to_remove_users,:) = [];
test_mask(to_remove_users,:) = [];
fprintf('%d more reviewers were removed b/c they did not have enough test and training reviews.\n',...
    sum(to_remove_users));

% remove beers that don't have any reviews in training set
to_remove_beers = (sum(~isnan(Xtrain),1) == 0) | ...
                  (sum(~isnan(Xtest),1) == 0);
Xtrain(:,to_remove_beers) = [];
Xtest(:,to_remove_beers) = [];
test_mask(:,to_remove_beers) = [];
fprintf('%d beers were removed because they did not have any training or testing data.\n',...
    sum(to_remove_beers));

%% --- CALCULATE TESTING BASELINE BEFORE CENTERING --- %%
[precenter_beer_avg_mse, precenter_user_avg_mse, precenter_global_mse, ...
    ~, ~, ~] = get_avg_baseline(Xtest);

[bias_avg_baseline, beer_avgs, user_bias] = bias_baseline(Xtrain, Xtest);
base_line_predictions = repmat(beer_avgs,length(user_bias),1) + ...
                        repmat(user_bias,1,length(beer_avgs));

%% ---- CENTER DATA ---- %%
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
elseif center_param == 5
    Xtrain = Xtrain - base_line_predictions;
else                                    % or globally
    fprintf('centering globally\n');
    Xtrain = center_globally(Xtrain);
    Xtest = center_globally(Xtest);
end

%% ---- INITIALIZATION OF U, V, C, D ---- %%
d = size(Xtrain,2);
n = size(Xtrain,1); 
init_lim = .8;
%initialize the parameter to some small random value     
svd.U = unifrnd( -init_lim, init_lim, [n, K]); 
svd.V = unifrnd( -init_lim, init_lim, [d, K]);
svd.c = unifrnd( -init_lim, init_lim, [n, 1]);
svd.d = unifrnd( -init_lim, init_lim, [1, d]);

%% ---- EXECUTE SVD ---- %%
    
params.max_iterations = 10;
iters = 2000;
training_error = [];
testing_error = zeros(iters,2);
total_iterations = 0;
for i = 1:iters
    
    if svd_mode == 1             
        % improved method with movie and user bias
        [mse, UV] = svd_train_improved(Xtrain,svd,params);
        
    elseif (regularization_param == 0)
        % basic SVD without regularization
        [mse, UV] = svd_train(Xtrain,svd,params);
        
    else
        % basic SVD
        [mse, UV] = svd_reg_train(Xtrain,svd,params,w);
    end
    
    total_iterations = total_iterations + length(mse);
    training_error = [training_error, mse];
    svd = UV;
    
    % calculate testing errors
%     testing_error(i,1) = total_iterations;
%     testing_error(i,2) = svd_testing_error(Xtest, UV.U, UV.V);
    
    if (length(mse) == 2)
        fprintf('stopping because training converged\n');
        break;
%     elseif ((i > 1) && (testing_error(i,2) > testing_error(i-1,2)))
%         fprintf('stopping because testing error increased\n')
%         break;
    end 
    fprintf('iteration: %d\n',i);
end

if i == iters
    fprintf('stopped because maximum iterations was reached\n')
end




%% ---- REPORT RESULTS ---- %%
% calculate baseline results
[bias_mse, baseline_bias_mse] = calc_bias_error(Xtest, beer_avgs,...
    user_bias, UV.U * UV.V');

% plot testing and training error
figure(); hold on;
plot(training_error,'r-');
%plot(testing_error(1:i,1),testing_error(1:i,2),'bo')
xlabel('iteration');
ylabel('MSE');
%legend('Training Error','Testing Error');
title('Training Error');

desc_str = sprintf(['n = %d, d = %d\nk = %d, sparcity = %.3f\n', ...
    'centering = %d; reg = %d,\n%d iterations\n|U| = %.1f, |V| = %.1f\n', ...
    'Training Error: %.4f\n'],...
    n, d, K, sparcity, center_param, regularization_param,...
    length(training_error), norm(UV.U), norm(UV.V), mse(end));
% desc_str = sprintf(['n = %d, d = %d\nk = %d, sparcity = %.3f\n', ...
%     'centering = %d; reg = %d,\n%d iterations\n|U| = %.1f, |V| = %.1f\n', ...
%     'Training Error: %.4f\nTesting Error = %.4f\n'],...
%     n, d, K, sparcity, center_param, regularization_param,...
%     length(training_error), norm(UV.U), norm(UV.V), mse(end), testing_error(i,2));
ylim=get(gca,'ylim');
xlim=get(gca,'xlim');
text(.35*(xlim(2)-xlim(1))+xlim(1),.6*(ylim(2)-ylim(1))+ylim(1),desc_str);

%% ---- CHANGE PRINTED MSE GRAPH NAME HERE ---- %%
print -dpng 'figures/7k_all_centered_on_avg_and_bias'

% record baselines (averages
[test_error, user_err_avgs, user_error_var] = ...
    svd_testing_error(Xtest - repmat(beer_avgs,size(Xtest,1),1) +...
        repmat(user_bias,1,size(Xtest,2)), UV.U, UV.V);

% calculate baselines 
test_reviews = sum(sum(~isnan(Xtest),1));
train_reviews = sum(sum(~isnan(Xtrain),1));
[beer_avg_mse, user_avg_mse, global_mse, ...
    ~, ~, ~] = get_avg_baseline(Xtest);

fprintf('--- REPORT ---\n');
fprintf('# training reviews: %d; # testing reviews: %d\n',train_reviews,test_reviews);
fprintf('Movie Avg and User Bias Baseline MSE: %.4f\n',bias_avg_baseline);
fprintf('Pre-Centering -- Beer avg MSE: %.4f, User avg MSE: %.4f, Global avg MSE: %.4f\n',...
    precenter_beer_avg_mse, precenter_user_avg_mse, precenter_global_mse);
fprintf('Testing error is %.4f. Beer avg MSE: %.4f, User avg MSE: %.4f, Global avg MSE: %.4f\n',...
    test_error, beer_avg_mse, user_avg_mse, global_mse);
fprintf('Bias Prediction Error: %.4f (Baseline at %.4f)\n',bias_mse,baseline_bias_mse);

figure();
subplot(1,2,1)
hist(user_err_avgs,100);
title('average errors for users')

subplot(1,2,2)
hist(user_error_var,100);
title('variance of errors for users')

% % Now try holdout error prediction (fresh users)
% holdout_error = svd_predict_and_test(Xholdout,UV.V);
% fprintf('Holdout error is %.4f\n',holdout_error)


