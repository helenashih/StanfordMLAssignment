function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
    C_values = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
    sigma_values = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

    cv_errors = zeros(length(C_values), length(sigma_values));

    % C_values outer loop, sigma_values inner loop
    for i = 1:length(C_values)
        for j = 1:length(sigma_values)
            C_cv = C_values(i);
            sigma_cv = sigma_values(j);
            cv_model = svmTrain(X, y, C_cv, @(x1, x2) gaussianKernel(x1, x2, sigma_cv));
            predictions = svmPredict(cv_model, Xval);
            prediction_errors(i, j) = mean(double(predictions ~= yval));
        end
    end
    % find the min predicted error row
    [error_values, row_id] = min(prediction_errors);
    [~ ,col] = min(error_values);
    row = row_id(col);
    C = C_values(row);
    sigma = sigma_values(col);	
% =========================================================================

end
