function betas = lasso(x, y, lambdas, n_it, acc)
%LASSO lasso regularization using co-ordinate descent optimization method.
%   betas = LASSO(lambdas, x, y, n_it) returns a matrix of lasso estimates
%   for each value of lambda.
%
%   x is the design matrix, i.e. x = [1, predictors].
%   y is the response matrix.
%   lambdas is the sequence of penalty term for the LASSO objective.
%   n_it is the maximum number of iteration for the update of coefficients.
%   acc is the accuracy of the co-ordinate descent method.
%
%   Example:
%       x = normrnd(0, 1, 500 ,20);
%       b = datasample(-5:2:5, 20)';
%       er = normrnd(0, 1, 500, 1);
%       y = x * b + er;
%       lambdas = exp(linspace(-5, 5, 100));
%
%       betas = lasso(lambdas, x, y, 100);

%% intial setups
% characterise x
x_p = size(x, 2);
x_n = size(x, 1);
% charecterise y
y_n = size(y, 1);

% defaults
if (nargin == 2)
    lmax = max(max(abs((x'*y)./diag(x'*x))), max(abs(x'*y)/500));
    lambdas = exp(linspace(-5, lmax, 100));
    n_it = 100;
    acc = (max(max(abs((x'*y)./diag(x'*x))), max(abs(x'*y)/500))) * 0.00001;
elseif (nargin == 3)
    n_it = 100;
    acc = (max(max(abs((x'*y)./diag(x'*x))), max(abs(x'*y)/500))) * 0.00001;
elseif (nargin ==4)
    acc = (max(max(abs((x'*y)./diag(x'*x))), max(abs(x'*y)/500))) * 0.00001;
end

% charecterise lambdas
l_n = size(lambdas, 2);

% error checking
if x_n ~= y_n
   error('dimension of inputs and output must be same')
end

%% LASSO
beta0 = zeros((x_p), 1);
betas = zeros((x_p), l_n);

for k = 1:l_n
    betas(:,(k)) = lasso_cd(lambdas(1,(k)), x, y, beta0, n_it, acc);
    beta0 = betas(:,k);
end

%% function for LASSO estimate
function x_best = lasso_cd(lambda, x, y, beta0, n_it, acc)
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
        value = x(:,i)' * (y - x(:,[1:(i - 1) (i + 1):size(x, 2)])...
            * beta([1:(i - 1) (i + 1):size(x, 2)])) / (x(:,i)' * x(:,i));
    end

    function z_best = cd_opt(z, f, v, s, n_it)
        fz = f(z);
        z_best = z;
        fz_best = fz;
        for j = 1:1:n_it
            z_last = z;
            for i = 1:1:size(z, 1)
                z(i) = s(v(i,z));
            end
            fz = f(z);
            if (fz < fz_best)
                z_best = z;
                fz_best = fz;
            end
            if (sum(abs(z_last - z)) < acc)
                z_best = z;
                break
            end
        end
    end
end
end
