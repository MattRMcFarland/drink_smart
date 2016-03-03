function [testing_error] = svd_testing_error(Xtest, U, V)
% INPUT
%   Xtest - n x d
%   U - n x k
%   V - k x d
%
% OUTPUT
%   testing_error - 1x1

test_mask = ~isnan(Xtest);
Xtest(isnan(Xtest)) = 0;
predictions = U * V';
user_errors = sum( (test_mask .* (Xtest - predictions)).^2,2) ./ ...
    sum(test_mask,2);
testing_error = mean(user_errors);
return;