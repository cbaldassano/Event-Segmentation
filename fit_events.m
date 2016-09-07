function eventModel = fit_events(trainingData, nEvents, varargin)
%Fits event segmentation model using annealed Baum-Welch
%
% trainingData (required): data for learning scene patterns
%    Voxels by Timepoints matrix, or cell array of multiple such matrices
%
% nScenes (required): number of scenes to find in training data
%
% Returns eventModel, containing:
%    loggamma: log p(event at time t = k)
%    loglikelihood: p(data|model), measure of model fit
%    eventPatterns: voxel x event matrix of learned event patterns
%    eventVar: variance of final model
%
% Optional arguments specified as property-value pairs:
%
% eventVariance: annealing function for Baum-Welch
%    Default is 4*0.98^(step-1)
%
% maxSteps: iteration limit, if no log-likelihood peak found
%    Default is 500
%
% scrambleCorrespond: when given two training sets, randomize the
%    correspondence between their scenes
%    Default is false


%% Input validation and standardization
inpParse = inputParser;
inpParse.FunctionName = 'fit_events';
inpParse.addParamValue('eventVariance',@(step) 4*0.98^(step-1));
inpParse.addParamValue('maxSteps',500);
inpParse.addParamValue('scrambleCorrespond',false);

inpParse.parse(varargin{:});
eventVariance = inpParse.Results.eventVariance;
maxSteps = inpParse.Results.maxSteps;
scrambleCorrespond = inpParse.Results.scrambleCorrespond;

if ~isa(eventVariance, 'function_handle')
    eventVariance = @(step) eventVariance;
end

if ~iscell(trainingData)
    trainingData = {trainingData};
end
nTrainSets = length(trainingData);

trainingSegments = cell(nTrainSets,1);
for i = 1:nTrainSets
    trainingSegments{i} = ones(size(trainingData{i},2),nEvents);
end

nDim = size(trainingData{1},1);
if (any(cellfun(@(x) size(x,1) ~= nDim, trainingData)))
    error('All input datasets must have the same number of voxels');
end

if (scrambleCorrespond)
    if (length(trainingData) < 2)
        error(['Can only scramble correspondences between ' ...
                'multiple training datasets']);
    end
    scram = zeros(length(trainingData),nScenes);
    for i = 1:length(trainingData)
        scram(i,:) = randperm(nScenes);
    end
end

%% Setting up transition matrix
% Stay or advance, with final sink state
Pi = [1 zeros(1,nEvents-1)];
P = [0.5*diag(ones(nEvents,1)) + 0.5*diag(ones(nEvents-1,1),1) ...
    [zeros(nEvents-1,1);0.5]];
EndPi = [zeros(1,nEvents-1) 1];

%% Main fitting loop
stepnum = 1;
ll_hist = zeros(0,length(trainingData));
end_ll = -Inf*ones(length(trainingData),1);

while (stepnum <= maxSteps)
    iterationVar = eventVariance(stepnum);

    % Compute mean patterns based on current segmentation
    meanPatterns = zeros(nTrainSets,nDim,nEvents);
    for i = 1:nTrainSets
        meanPatterns(i,:,:) = trainingData{i} * ...
          bsxfun(@times,trainingSegments{i},1./sum(trainingSegments{i},1));
    end
    meanPatterns = squeeze(mean(meanPatterns,1));

    LL = zeros(length(trainingData),1);
    loggamma = cell(length(trainingData),1);
    
    % Fit segmentation based on mean patterns
    for i = 1:length(trainingData)
        if (scrambleCorrespond)
            logprob = logprob_obs(trainingData{i}, ...
                               meanPatterns(:,scram(i,:)), iterationVar);
        else
            logprob = logprob_obs(trainingData{i}, ...
                               meanPatterns, iterationVar);
        end
        [loggamma{i}, LL(i)] = forward_backward_log(logprob, Pi, EndPi, P);
    end
    
    if (scrambleCorrespond)
        for i = 1:nTrainSets
            loggamma{i}(:,scram(i,:)) = loggamma{i};
        end
    end

    % Break after finding maximum log-likelihood
    ll_hist = [ll_hist; LL'];
    if (mean(LL) > mean(end_ll))
        end_ll = LL;
        end_scale = iterationVar;
        end_loggamma = loggamma;
    end
    if (mean(end_ll)-mean(LL) > 1)
        % LL has started to decrease
        break;
    end
    
    % Update segmentation for next round of BW
    trainingSegments = cellfun(@exp, loggamma, 'UniformOutput', false);
    stepnum = stepnum + 1;
end

%% Output packaging
if (length(end_loggamma)==1)
    end_loggamma = end_loggamma{1};
end
eventModel.loggamma = end_loggamma;
eventModel.loglikelihood = end_ll;
eventModel.eventPatterns = meanPatterns;
eventModel.eventVar = end_scale;

end
