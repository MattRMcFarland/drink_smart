function [Xtrain, Xtest] = get_test_points(X, test_pcent)
%% randomly selects test_pcent of points for testing

% get test points
test_mask = ~isnan(X) .* (rand(size(X,1),size(X,2)) < test_pcent);
Xtrain = ~test_mask .* X;
Xtrain(test_mask == 1) = NaN;           % mark testing spots as untried
Xtest = test_mask .* X;
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

end