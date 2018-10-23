function betas = lasso(lambdas, x, y, n_it)
%LASSO lasso regularization using co-ordinate descent optimization method.
%   betas = LASSO(lambdas, x, y, n_it) returns a matrix of lasso estimates
%   for each value of lambda.
%
%   lambdas is the sequence of penalty term for the LASSO objective.
%   x is the predictor matrix.
%   y is the response matrix.
%   n_it is the maximum number of iteration for the update of coefficients.
%
%   Example:
%       x = normrnd(0, 1, 500 ,20);
%       b = datasample(-5:2:5, 20)';
%       er = normrnd(0, 1, 500, 1);
%       y = x * b + er;
%       lambdas = exp(linspace(-5, 5, 100));
%
%       betas = lasso(lambdas, x, y, 100);

beta0 = ones(20, 1);
betas = zeros(20, size(lambdas, 2));
k = 1;
sx = size(x, 2);

while (sum(abs(beta0)) > 0.00001)
    k = k + 1;
    betas(:,(k - 1)) = lasso_cd(lambdas(1,(k - 1)), x, y, beta0, n_it);
    beta0 = betas(:,(k-1));
end

function x_best = lasso_cd(lambda, x, y, beta0, n_it)

soft = @(lambda)(@(x)(sign(x) * max(0, abs(x) - lambda)));
s = soft(lambda);
f = @(beta)(lasso_f(lambda, x, y, beta));
v = @(i, beta)st_f(i, x, y, beta);

x_best = cd_opt(beta0, f, v, s, n_it);


% functions required for evaluation of the LASSO estimates

    % the LASSO objective function
    function value = lasso_f (lambda, x, y, beta)
        value = sum((y - x * beta) .^ 2) / (2 * size(x , 1))...
            + lambda * sum(abs(beta));
    end
    
    % soft thresholding function for co-ordinate descent method
    function value = st_f (i, x, y, beta)
        value = x(:,i)' * (y - x(:,[1:(i - 1) (i + 1):sx]) * ...
            beta([1:(i - 1) (i + 1):sx])) / (x(:,i)' * x(:,i));
    end

    function x_best = cd_opt(x, f, v, s, n_it)
        fx = f(x);
        x_best = x;
        fx_best = fx;
        for j = 1:1:n_it
            x_last = x;
            for i = 1:1:size(x, 1)
                x(i) = s(v(i,x));
            end
            fx = f(x);
            if (fx < fx_best)
                x_best = x;
                fx_best = fx;
            end
            if (sum(abs(x_last - x)) < 0.0001)
                x_best = x;
                break
            end
        end
    end
end
end
