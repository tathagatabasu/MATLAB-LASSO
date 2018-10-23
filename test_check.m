% Test-check
      
x = normrnd(0, 1, 500 ,20);
b = datasample(-5:2:5, 20)';
er = normrnd(0, 1, 500, 1);
y = x * b + er;
lambdas = exp(linspace(-5, 5, 100));
betas = lasso(lambdas, x, y, 100);
      
figure
plot(log(lambdas), betas)

[coef, summ] = cv_lasso(lambdas, x, y, 5, 100, 10);