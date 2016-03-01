function  [dw, dv] = numerical_gradient(fun, svd, epsilon)
% evaluate the numerical gradient of function fun at theta
%

if nargin < 3, epsilon = 1e-5; end

dw = zeros( size(svd.U) ); 
dv = zeros( size(svd.V) );

for i = 1:length(svd.U(:))
    net2 = svd; 
    net2.U(i) = svd.U(i) + 1e-5;
    
    dw(i) = (fun(net2)-fun(svd)) / epsilon;
end

for i = 1:length(svd.V(:))
    net2 = svd; 
    net2.V(i) = svd.V(i) + 1e-5;
    
    dv(i) = (fun(net2)-fun(svd)) / epsilon;
end