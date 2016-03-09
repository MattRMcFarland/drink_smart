%% load the training and testing dataset
close all; clear all;

d = importdata('data/small_collected_reviews.csv',',');
X = d.data;

n = size(X, 1);
d = size(X, 2);
K = 5;

% limit size of dataset so it doesn't take forever;
small_n = 50;
if small_n > n
    small_n = n
end
small_d = 50;
if small_d > d
    small_d = d
end

% redefine X so it doesn't have any missing values
X = unifrnd( -.5, .5, [small_n, small_d]);

%% Checking your gradient: compare your gradient with the numerical gradient
perturb = -5:1:5;
grd = []; grd2 = [];
epsilon = 1e-5;
for i = 1:length(perturb)
    % to make sure your gradient is correct,you can compare your gradient 
    % calculating with numerical gradient
    %small_i = randperm(size(X,1), small_n);
    %small_x = X(small_i, 1:small_d); 
    
    % randomly initialize U and V
    svd.U = unifrnd( -.5, .5, [small_n, K]);  
    svd.V = unifrnd( -.5, .5, [small_d, K]);
    
    % get gradient
    [mse, grad] = svd_error(X, svd); 
    grd = [grd; [grad.U(:)', grad.V(:)']];
    
    % get numerical gradient
    fun=@(th)(svd_error(X, th));
    [du, dv] = numerical_gradient(fun, svd);  
    grd2 = [grd2; [du(:)', dv(:)']]; % evaluate the numerical gradient
    fprintf('Iteration: %d\n',i);
end
figure; hold on;
%plotyy(perturb, mean(grd.^2,2),perturb, mean(grd2.^2,2))
plot(perturb, mean(grd.^2,2),'-ro'); % plot the L2 norm of true gradient
plot(perturb, mean(grd2.^2,2), '--*'); % the L2 norm of the numerical gradient
legend('calculated gradient','numerical gradient')
title('checking gradient')
xlabel('perturb')
ylabel('L2 norm of gradient')

print -dpng figures/check_grad_descent
