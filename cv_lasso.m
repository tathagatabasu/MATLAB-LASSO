function [coef, summary] = cv_lasso(lambdas, x, y, k, n_it, df)
%CV_LASSO cross-validated LASSO estimates
%   [coef, summary] = CV_LASSO(lambdas, x, y, k, n_it, df) returns a 
%   structure of coefficients and summary frame of mean-squared error with 
%   degrees of freedom.
%
%   lambdas is the sequence of penalty term for the LASSO objective.
%   x is the predictor matrix.
%   y is the response matrix.
%   k is the number of fold for cross-validation.
%   n_it is the maximum number of iteration for the update of coefficients.
%   df is the minimum degree of freedom. df allows user to reduce 
%   the number of variables in regression coefficients. df = size(x, 2) 
%   will make every estimate equal to zero.
%
%   Example:
%       x = normrnd(0, 1, 500 ,20);
%       b = datasample(-5:2:5, 20)';
%       er = normrnd(0, 1, 500, 1);
%       y = x * b + er;
%       lambdas = exp(linspace(-5, 5, 100));
%
%       [coef, summ] = cv_lasso(lambdas, x, y, 5, 100, 10);

data_partition = cv_random_partition(x, y, k)';

model = arrayfun(@(i)cv_train(i, data_partition, lambdas, n_it), 1:k,...
    'UniformOutput', false);

% storing the coefficients obtained from k-fold cross-validation
betas_store = zeros(size(x, 2), size(lambdas, 2), k);
for i = 1:k
    betas_store(:,:,i) = cell2mat(model(i));
end
betas = mean(betas_store, 3);

% error-handling
error = arrayfun(@(i)cv_test(i, data_partition, model'), 1:k,...
    'UniformOutput', false);
error_mean = mean(cell2mat(error'), 1);
error_sd = std(cell2mat(error'), 0, 1);

% degrees of freedom
dfs = arrayfun(@(i)nnz(betas(:,i) == 0), 1:size(lambdas, 2));

%%% LASSO estimates
% if df = 0, then used 
error_min_ind = find(error_mean == min(error_mean), 1, 'last');
beta = betas(:,error_min_ind);
cv_error = error_mean(error_min_ind);

% if df > 0, then used
if (df > 0)
    df_inds = find(dfs >= df);
    df_ind = df_inds(find(error_mean(df_inds) == ...
        min(error_mean(df_inds)), 1, 'last'));
    beta = betas(:, df_ind);
    cv_error = error_mean(df_ind);
end

% intercept
intercept = mean(y, 1) - mean(x, 1) * beta;

%%% Figures
stop_ind = find(sum(abs(betas)) == 0, 1 );
% cross-validation curve
figure
errorbar(log(lambdas(1:stop_ind)), error_mean(1:stop_ind),...
    error_sd(1:stop_ind),'-s', 'MarkerSize', 2,...
    'MarkerEdgeColor', 'red', 'MarkerFaceColor', 'red')
title('cross-validation curve')
xlabel('log(\lambda)')
ylabel('mean-squared error')

% LASSO coefficient path
figure
plot(log(lambdas(1:stop_ind)), betas(:,1:stop_ind))
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
function model = cv_train(i, data_partition, lambdas, n_it)

traindata = data_partition;
traindata(i,:) = [];
traindata = cell2mat(traindata);
x = traindata(:,2:size(traindata, 2));
y = traindata(:,1);
model = lasso(lambdas, x, y, n_it);
end
% function for testing the model
function error = cv_test(i, data_partition, model)

testdata = data_partition(i,:);
testdata = cell2mat(testdata);
x = testdata(:,2:size(testdata, 2));
y = testdata(:,1);
model = cell2mat(model(i,:));
cv_predict = @(beta, x)(x * beta);
estimate = arrayfun(@(j)(cv_predict(model(:,j), x)), 1:size(model, 2),...
    'UniformOutput', false);
estimate = cell2mat(estimate);
error = arrayfun(@(j)cv_mse(y, estimate(:,j)), 1:size(estimate, 2));
end

%%% output
% coef
coef.('intercept') = intercept;
for k = 2:21
    name = sprintf('var%d', (k - 1));
    coef.(name) = beta((k - 1));
end
coef.('df') = nnz(beta == 0);
coef.('mse') = cv_error;
% error
summary = table;
summary.lambda = lambdas(1:stop_ind)';
summary.mse = error_mean(1:stop_ind)';
summary.df = dfs(1:stop_ind)';
end
