function [coef, summary] = cv_lasso(x, y, df, k, n_it, acc)
%CV_LASSO cross-validated LASSO estimates
%   [coef, summary] = CV_LASSO(lambdas, x, y, df, k, n_it) returns a 
%   structure of coefficients and summary frame of mean-squared error with 
%   degrees of freedom.
%
%   x is the predictor matrix.
%   y is the response matrix.
%   df is the number of non-zero variables. df allows user to reduce 
%   the number of predictors in the output. Default value is size(x, 2).
%   k is the number of fold for cross-validation. Default value is 5
%   n_it is the maximum number of iteration for the update of coefficients.
%   Default value is 100.
%   acc is the accuracy of the co-ordinate descent method. Default value is
%   0.00001
%
%   Example:
%       x = normrnd(0, 1, 500 ,20);
%       b = datasample(-5:2:5, 20)';
%       er = normrnd(0, 1, 500, 1);
%       y = x * b + er;
%       lambdas = exp(linspace(-5, 5, 100));
%
%       [coef, summ] = cv_lasso(lambdas, x, y, 5, 100, 10);
 
% characterise x
x_p = size(x, 2);
x_n = size(x, 1);
% charecterise y
y_n = size(y, 1);

% defaults
if (nargin == 2)
    df = x_p;
    k = 5;
    n_it = 100;
    acc = 0.00001;
elseif (nargin == 3)
    k = 5;
    n_it = 100;
    acc = 0.00001;
elseif (nargin == 4)
    n_it = 100;
    acc = 0.00001;
elseif (nargin == 5)
    acc = 0.00001;
end


% error checking
if x_n ~= y_n
   error('dimension of inputs and output must be same')
end
if ((df < 0) || (df > x_p))
    error('degrees of freedom must be between 0 and no. of predictors')
end
if k < 2
    error('number of folds must be greater than or equal to 2')
end
% rounding non-integer inputs
if df ~= round(df)
    fprintf("degrees of freedom rounded to nearest integer %d \n",...
        round(df))
    df = round(df);
end
if k ~= round(k)
    fprintf("number of folds rounded to nearest integer %d \n", round(k))
    k = round(k);
end
if n_it ~= round(n_it)
    fprintf("number of iterations rounded to nearest integer %d \n",...
        round(n_it))
    n_it = round(n_it);
end

lmax = max(max(abs((x'*y)./diag(x'*x))), max(abs(x'*y)/500));
lambdas = exp(linspace(-5, lmax, 100));
z = [ones(x_n, 1), x];
% charecterise lambdas
l_n = size(lambdas, 2);

data_partition = cv_random_partition(z, y, k)';

model = arrayfun(@(i)cv_train(i, data_partition, lambdas, n_it, acc), 1:k,...
    'UniformOutput', false);

% storing the coefficients obtained from k-fold cross-validation
betas_store = zeros((x_p+1), l_n, k);
for i = 1:k
    betas_store(:,:,i) = cell2mat(model(i));
end
betas = mean(betas_store, 3);

% error-handling
model_error = arrayfun(@(i)cv_test(i, data_partition, model'), 1:k,...
    'UniformOutput', false);
error_mean = mean(cell2mat(model_error'), 1);
error_sd = std(cell2mat(model_error'), 0, 1);

% degrees of freedom
dfs = arrayfun(@(i)nnz(betas(2:(x_p+1),i) ~= 0), 1:l_n);

%%% LASSO estimates

% if df < size(x, 2), then used
if (df < x_p)
    df_inds = find(dfs <= df);
    df_ind = df_inds(find(error_mean(df_inds) == ...
        min(error_mean(df_inds)), 1, 'last'));
    beta = betas(:, df_ind);
    cv_error = error_mean(df_ind);
else % if df = size(x, 2), then used 
    df_ind = find(error_mean == min(error_mean), 1, 'last');
    beta = betas(:,df_ind);
    cv_error = error_mean(df_ind);
end

%%% Figures
stop_ind = find(sum(abs(betas)) == 0, 1 );
% cross-validation curve
figure
errorbar(log(lambdas(1:stop_ind)), error_mean(1:stop_ind),...
    error_sd(1:stop_ind),'-s', 'MarkerSize', 2,...
    'MarkerEdgeColor', 'red', 'MarkerFaceColor', 'red')
yax = line([log(lambdas(df_ind)), log(lambdas(df_ind))], ylim);
yax.Color = 'k';
title('cross-validation curve')
xlabel('log(\lambda)')
ylabel('mean-squared error')

% LASSO coefficient path
figure
plot(log(lambdas(1:stop_ind)), betas(2:(x_p+1),1:stop_ind))
xax = refline([0 0]);
xax.Color = 'k';
yax = line([log(lambdas(df_ind)), log(lambdas(df_ind))], ylim);
yax.Color = 'k';
title('lasso coefficient path')
xlabel('log(\lambda)')
ylabel('values of \beta')

%%% Functions required for evaluation of cross-validated LASSO estimates
% function for creating k-fold random partition for cross-validation
function split_data = cv_random_partition(x, y, k)

cv_data = [y x];
cv_data = cv_data(1:(fix((size(cv_data, 1) / k)) * k),:);

index_sample = datasample(1:size(cv_data, 1), size(cv_data, 1),...
    'Replace', false);
cv_data = cv_data(index_sample,:);
div = fix(size(cv_data, 1) / k);

split_data = arrayfun(@(i)cv_data(((i - 1) * div + 1):(i * div),:),...
    1:k, 'UniformOutput', false);
end
% function for calculating mean-squared error
function error = cv_mse(original, estimate)

difference = original - estimate;
difference_square = difference .^ 2;
total_error = sum(difference_square);
error = total_error / size(estimate, 1);
end
% function for trainig the model
function model = cv_train(i, data_partition, lambdas, n_it, acc)

traindata = data_partition;
traindata(i,:) = [];
traindata = cell2mat(traindata);
x = traindata(:,2:size(traindata, 2));
y = traindata(:,1);
model = lasso(lambdas, x, y, n_it, acc);
end
% function for testing the model
function test_error = cv_test(i, data_partition, model)

testdata = data_partition(i,:);
testdata = cell2mat(testdata);
x = testdata(:,2:size(testdata, 2));
y = testdata(:,1);
model = cell2mat(model(i,:));
cv_predict = @(beta, x)(x * beta);
estimate = arrayfun(@(j)(cv_predict(model(:,j), x)), 1:size(model, 2),...
    'UniformOutput', false);
estimate = cell2mat(estimate);
test_error = arrayfun(@(j)cv_mse(y, estimate(:,j)), 1:size(estimate, 2));
end

%%% output
% coef
coef.('intercept') = beta((1));
for k = 2:size(x, 2)
    name = sprintf('beta%d', (k - 1));
    coef.(name) = beta((k));
end
coef.('df') = x_p - nnz(beta == 0);
coef.('mse') = cv_error;
% error
summary = table;
summary.lambda = lambdas(1:stop_ind)';
summary.mse = error_mean(1:stop_ind)';
summary.df = dfs(1:stop_ind)';
end
