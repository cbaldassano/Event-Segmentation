function [loggamma, loglikelihood] = find_events(eventPatterns, eventVar, testingData)
% Given event patterns, find matching events in a new dataset
%
% eventPatterns: voxel x event matrix of event patterns
%    This can come from eventModel.eventPatterns or another source
%    To generate null fits, supply eventPatterns with scrambled event order
%
% eventVar: scalar, or 1 x event vector of event-specific variances
%    Can come from eventModel.eventVar or another source
%
% testingData: voxel x timepoint matrix of fMRI data
%
% Returns:
%    loggamma: log p(event at time t = k)
%    loglikelihood: p(data|model), measure of model fit

%% Setting up transition matrix
nEvents = size(eventPatterns,2);
% Stay or advance, with final sink state
Pi = [1 zeros(1,nEvents-1)];
P = [0.5*diag(ones(nEvents,1)) + 0.5*diag(ones(nEvents-1,1),1) ...
    [zeros(nEvents-1,1);0.5]];
EndPi = [zeros(1,nEvents-1) 1];

%% Find events
logprob = logprob_obs(testingData, eventPatterns, eventVar);
[loggamma, loglikelihood] = forward_backward_log(logprob, Pi, EndPi, P);

end
