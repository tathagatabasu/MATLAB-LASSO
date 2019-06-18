% Test-check
clc     
x = normrnd(0, 1, 500 ,20);
b = datasample(-4:2:4, 20)';
er = normrnd(0, 1, 500, 1);
y = 5 * ones(size(x, 1),1) + x * b + er;
lambdas = exp(linspace(-5, 5, 100));
betas = lasso(lambdas, x, y, 100, 0.00001);
      
figure
plot(log(lambdas), betas)

[coef, summ] = cv_lasso(x, y, 15, 10);