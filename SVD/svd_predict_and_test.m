function [mse] = svd_predict_and_test(Xtest, V)
% INPUTS
%   Xtest   - n x d
%   V       - K x d
%
% OUTPUTS
%   mse     - 1 x 1 

test_I = ~isnan(Xtest);
Xtest(isnan(Xtest)) = 0;
U_test = test_I .* Xtest * V / (V' * V);
test_predictions = U_test * V';
test_counts = sum(test_I,2);
user_errors = sum( test_I .* (Xtest - test_predictions).^2,2) ./ test_counts;
mse = mean(user_errors);
end