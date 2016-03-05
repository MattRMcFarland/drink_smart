function [testing_error, avg_user_mse, avg_user_mse_stdevs] = svd_testing_error(Xtest, U, V)
% INPUT
%   Xtest - n x d
%   U - n x k
%   V - k x d
%
% OUTPUT
%   testing_error - 1x1
%   avg_user_mse = n x 1
%   avg_user_mse_stdevs = n x 1

n = size(Xtest,1);
d = size(Xtest,2);

test_mask = ~isnan(Xtest);
Xtest(isnan(Xtest)) = 0;
predictions = U * V';
%user_errors = sum( (test_mask .* (Xtest - predictions)).^2,2) ./ ...
%    sum(test_mask,2);
%testing_error = mean(user_errors);
testing_error = sum(sum(test_mask .* (Xtest - predictions).^2,2)) ./ ...
    sum(sum(test_mask,2));

avg_user_mse = sum(test_mask .* (Xtest - predictions).^2,2) ./ ...
    sum(test_mask,2);

avg_user_mse_stdevs = sum((test_mask .* (Xtest - predictions).^2 - ...
    repmat(avg_user_mse,1,d)).^2,2) ./ (sum(test_mask,2) - 1);

return;