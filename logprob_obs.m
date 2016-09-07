function logprob = logprob_obs(targetData, meanPat, eventVar)
% Compute log probability of targetData being generated from meanPat
% targetData: voxel x timepoint matrix
% meanPat: voxel x event matrix
% eventVar: scalar, or 1 x event vector of event-specific variances

nDim = size(meanPat,1);
K = size(meanPat,2);

targetData = zscore(targetData);
meanPat = zscore(meanPat);

logprob = zeros(size(targetData,2), K);

if length(eventVar) == 1
    eventVar = eventVar*ones(K,1);
end

for k = 1:K
    % Assumes isotropic variance for all features
    logprob(:,k) = -0.5 * nDim*log(2*pi*eventVar(k))-...
        0.5*sum(bsxfun(@minus,targetData',meanPat(:,k)').^2,2)/eventVar(k);
end

% Normalize dimension (equiv to scaling by 1/sqrt(nDim))
logprob = logprob/nDim;
end