function err_xVar = cross_validation(Xtrain, Nfold, train_func, evaluate_func, K)
% Cross validation: 
% Inputs: 
%    --- Xtrain (nXd),: training data
%    --- Nfold: number of partitions in cross validation
%    --- train_func: a function handle that takes Xtrain and "params" as inputs and
%    ouput estimated parameter \hat\theta
%   --- predict_func: a function handle that takes (Xtest and parameter \hat\theta as
%   inputs and outputs a testing error 
% Output: 
%    --- err_xVar: (1X1) the averaged cross validation error.  
errors = size(1,Nfold);

init_lim = 1.5;
for i = 1:Nfold              % N partitions
    [miniXtrain, miniXtest] = get_test_points(Xtrain, 1/Nfold);
    
    n = size(miniXtrain,1);
    d = size(miniXtrain,2);
    init_svd.U = unifrnd( -init_lim, init_lim, [n, K]); 
    init_svd.V = unifrnd( -init_lim, init_lim, [d, K]);
    
    % train on all data except testing testing
    [~,fold_svd]=train_func(miniXtrain, init_svd);
    
    % test on held data
    errors(i)=evaluate_func(miniXtest, fold_svd);
end     
err_xVar = mean(errors); % report average errors

end