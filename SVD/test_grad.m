%% testing SVD analysis

n = 100;
d = 50;
K = 10; 

X = unifrnd(-1, 1, [n d]);

% use svd_error to follow gradient to replicate X
X1 = X;
iteration = 1000;
step_size = 1e-2;

svd.U = unifrnd( -.5, .5, [n, K]); 
svd.V = unifrnd( -.5, .5, [d, K]);

training_error = [];
for i = 1:iteration
    [mse, grad] = svd_error(X1, svd);
    training_error = [training_error, mse];
    
    svd.U = svd.U - step_size * grad.U;
    svd.V = svd.V - step_size * grad.V;
    
    if (length(mse) > 1 && ...
        mse(end) - mse(end-1) < threshold)
        fprintf('below threshold at iteration %d\n',i);
        break;
    else
        fprintf('interation %d (mse: %.4f)\n',i,mse);
    end

end

X1_error = sum(sum((X - svd.U * svd.V').^2,2)) ./ (n * d)
figure()
plot(1:i,training_error)
xlabel('iteration')
ylabel('MSE')
title('Error Descent')

[U2,S2,V2] = svds(X,K);
X2_error = sum(sum((X - U2 * S2 * V2').^2,2)) ./ (n * d)

X1_X2_diff = sum(sum((svd.U * svd.V' - U2 * S2 * V2').^2,2)) ./ (n * d)
% then use svds() on X to get real U and V.