% Test-check
      
x = normrnd(0, 1, 500 ,20);
b = datasample(-4:2:4, 20)';
er = normrnd(5, 1, 500, 1);
y = x * b + er;
lambdas = exp(linspace(-5, 5, 100));
betas = lasso(lambdas, x, y, 100);
      
%figure
plot(log(lambdas), betas)

[coef, summ] = cv_lasso(x, y, 5, 100, 10);

x.p = size(x, 2);
x.n = size(x, 1);