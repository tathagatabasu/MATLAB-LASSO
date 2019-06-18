% Test-check
clc     
x = normrnd(0, 1, 500 ,20);
b = datasample(-4:2:4, 20)';
er = normrnd(0, 1, 500, 1);
y = 5 * ones(size(x, 1),1) + x * b + er;
lambdas = exp(linspace(-5, 10, 100));
betas = lasso(x, y);
      
figure
plot(log(lambdas), betas)

[coef, summ] = cv_lasso(x, y, 15, 10, 20, 0.0000001);