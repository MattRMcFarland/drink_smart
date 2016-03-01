%% load similarity matrix and test predictions
close all; clear all;
%sim = csvread('similarity_matrix.csv',2,2);

% test1.mat includes a testing matrix of users x items 
load test1.mat


avg_errors = zeros(size(testing,1),1);
for u = 1:size(testing,1)
    I = find(testing(u,:));  % find rated beers
    if length(I) < 1
        fprintf('user at index %d has no ratings for any beers\n',u);
        continue;
    else 
        user_errors = zeros(length(I),1);
        for i = 1:length(I)
            ratings = testing(u,:);
            ratings(I(i)) = 0;      % set 1 beer aside (will predict)
            predictions = sim * ratings';
            user_errors(i) = (predictions(i) - testing(u,i))^2; 
        end
        avg_errors(u) = mean(user_errors);
    end
end

avg_overall_mse = mean(avg_errors)
avg_std_mse = std(avg_errors)
    