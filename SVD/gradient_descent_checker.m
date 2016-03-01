%% load the training and testing dataset
close all; clear all;

d = importdata('small_collected_reviews.csv',',');
X = d.data;

n = size(X, 1)
d = size(X, 2)
K = 5

small_n = 15;
if small_n < n
    small_n = n;
end

%% Checking your gradient: compare your gradient with the numerical gradient
perturb = -5:1:5;
grd = []; grd2 = [];
for i = 1:length(perturb)
    % to make sure your gradient is correct,you can compare your gradient calculating with numerical gradient
    small_i = randperm(size(X,1), small_n);
    small_x = X(small_i, :); 
    %Ycheck = Ytrain(rndIdx);

    fun=@(th)(svd_error(small_x, th)); 
    svd.U = unifrnd( -6, 6, [small_n, K])  
    svd.V = unifrnd( -6, 6, [d, K])
    
    [mse, grad] = fun(svd); 
    grd = [grd; [grad.U(:)', grad.V(:)']];
    
    [du, dv] = numerical_gradient(fun, svd);  
    grd2 = [grd2; [du(:)', dv(:)']]; % evaluate the numerical gradient
    i
end
figure; hold on;
plot(perturb, mean(grd.^2,2),'-ro'); % plot the L2 norm of true gradient
plot(perturb, mean(grd2.^2,2), '--*'); % the L2 norm of the numerical gradient
legend('calculated gradient','numerical gradient')
title('checking gradient')
xlabel('perturb')
ylabel('L2 norm of gradient')

print -dpng figures/check_grad_descent
%*****If you code is correct, you should see that grd and grd2 matches closely
%with each other, and both have minimum L2 norm when perturb = 0. ******%
