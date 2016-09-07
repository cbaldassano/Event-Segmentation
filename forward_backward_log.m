function [loggamma, LL] = forward_backward_log(logprob, Pi, EndPi, P)
% Computes log p(event at time t = k), and log likelihood of fit
% Runs modified forward-backward algorithm that allows for ending event
% constraing (EndPi)
% Inputs:
%   logprob: time by event matrix of log p(response t | event k)
%   Pi, EndPi: 1 by event vector of starting and ending probs
%   P: state by state transition matrix, P(i,j) = prob transitioning from
%      event i to j. sum(P,2) should be all ones
%      If P has an extra column, this is treated as a dummy absorbing event
%      that is never observed
%
% Outputs:
%   loggamma: time by event matrix of posterior probabilities
%   LL: log-likelihood of fit, for model comparison


if (size(P,2) == size(P,1) + 1)
    P = [P; [zeros(1,size(P,1)) 1]];
    Pi = [Pi 0];
    EndPi = [EndPi 0];
    logprob = [logprob -Inf(size(logprob,1),1)];
    trim_gamma = true;
else
    trim_gamma = false;
end

T = size(logprob,1);
K = size(logprob,2);

% Forward pass
logscale=zeros(T,1);   % Prevents underflow, and used to compute LL
logalpha=zeros(T,K);
logbeta=zeros(T,K);
for t=1:T
  if (t==1)
    logalpha(1,:) = log(Pi) + logprob(1,:);
  else
    logalpha(t,:)=log(exp(logalpha(t-1,:))*P) + logprob(t,:);
  end
  logscale(t)=logsumexp(logalpha(t,:));
  logalpha(t,:)=logalpha(t,:) - logscale(t);
end

% Backward pass
logbeta(T,:)=log(EndPi) - logscale(T);
for t=T-1:-1:1
  obs_weighted = logbeta(t+1,:) + logprob(t+1,:);
  offset = max(obs_weighted);
  logbeta(t,:)=offset + log(exp(obs_weighted-offset)*(P')) - logscale(t); 
end

loggamma = logalpha + logbeta; 
loggamma=bsxfun(@minus, loggamma,logsumexp(loggamma,2));

LL = sum(logscale(1:(T-1))) + ...
     logsumexp(logalpha(T,:) + logscale(T) + log(EndPi));

if (trim_gamma)
    loggamma = loggamma(:,1:(end-1));
end
end

function LSE = logsumexp(x, dim)
% Compute log(sum(exp(x))) for very negative x
if nargin < 2
    [~,dim] = find(size(x)-1,1);
end
offset = max(x, [], dim);
LSE = offset + log(sum(exp(bsxfun(@minus,x,offset)), dim));
end