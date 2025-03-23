function [regML, validationRMSE] = regML(trainingData)
%  Input:
%       A table containing the same predictor and response
%       columns as those used in the model
%  Output:
%      regML: A struct containing the trained regression model.
%
%      trainedModel.predictFcn: A function to make predictions on new data.
%
%      validationRMSE: A double representing the validation RMSE.
%
% To make predictions with the returned 'trainedModel' on new data "T", use
%   yfit = regML.predictFcn(T)
%
% T must be a table containing at least the same predictor columns as used
% during training. For details, enter:
%   regML.HowToPredict

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;
predictorNames = {'IM1', 'IM2', 'IM6', 'IM8', 'IM13'};
predictors = inputTable(:, predictorNames);
response = inputTable.Pi_th;
isCategoricalPredictor = [false, false, false, false, false];

%Select subset of the features
includedPredictorNames = predictors.Properties.VariableNames([true true true true true]);
predictors = predictors(:,includedPredictorNames);
isCategoricalPredictor = isCategoricalPredictor([true true true true true]);

% Train a regression model
template = templateTree(...
    'MinLeafSize', 1, ...
    'NumVariablesToSample', 'all');
regressionEnsemble = fitrensemble(...
    predictors, ...
    response, ...
    'Method', 'Bag', ...
    'NumLearningCycles', 399, ...
    'Learners', template);

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
featureSelectionFcn = @(x) x(:,includedPredictorNames);
ensemblePredictFcn = @(x) predict(regressionEnsemble, x);
regML.predictFcn = @(x) ensemblePredictFcn(featureSelectionFcn(predictorExtractionFcn(x)));

% Add additional fields to the result struct
regML.RequiredVariables = {'IM1', 'IM2', 'IM6', 'IM8', 'IM13'};
regML.RegressionEnsemble = regressionEnsemble;
regML.About = 'This struct is a trained model exported from Regression Learner R2024a.';
regML.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appregression_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;
predictorNames = {'IM1', 'IM2', 'IM6', 'IM8', 'IM13'};
predictors = inputTable(:, predictorNames);
response = inputTable.Pi_th;
isCategoricalPredictor = [false, false, false, false false];

% Perform cross-validation
partitionedModel = crossval(regML.RegressionEnsemble, 'KFold', 10);

% Compute validation predictions
validationPredictions = kfoldPredict(partitionedModel);

% Compute validation RMSE
validationRMSE = sqrt(kfoldLoss(partitionedModel, 'LossFun', 'mse'));
