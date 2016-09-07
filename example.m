function example()
% Example of event segmentation and finding corresponding events

% Parameters for creating small simulated datasets
V = 10;
K = 10;
T = 500;
T2 = 300;

% Generate the first dataset
rng(1);
eventMeans = randn(V,K);
eventLabels = generate_event_labels(T, K, 0.1);
simulData = generate_data(V, T, eventLabels, eventMeans, 1);

% Find the events in this dataset
eventModel = fit_events(simulData, K);

% Generate other datasets with the same underlying sequence of event
% patterns, and try to find matching events
testLoops = 10;
boundMatch = zeros(2, testLoops);
LL = zeros(2, testLoops);
for test_i = 1:testLoops
    % Generate data
    eventLabels2 = generate_event_labels(T2, K, 0.5);
    simulData2 = generate_data(V, T2, eventLabels2, eventMeans, 0.1);

    % Find events matching previously-learned events
    [loggamma, LL(1, test_i)] = find_events(eventModel.eventPatterns, ...
                                            eventModel.eventVar, ...
                                            simulData2);
    [~, estEvents] = max(loggamma, [], 2);
    boundMatch(1, test_i) = 1 - sum(abs(diff(eventLabels2) - ...
                                        diff(estEvents'))) / (2 * K);

    % Run again, but with the order of events shuffled so that it no longer
    % corresponds to the training data
    scrambledEvents = eventModel.eventPatterns(:, randperm(K));
    [loggamma, LL(2, test_i)] = find_events(scrambledEvents, ...
                                            eventModel.eventVar, ...
                                            simulData2);
    [~, estEvents] = max(loggamma, [], 2);
    boundMatch(2, test_i) = 1 - sum(abs(diff(eventLabels2) - ...
                                        diff(estEvents'))) / (2 * K);
end

% Across the testing datasets, print how well we identify the true event
% boundaries and the log-likehoods in real vs. shuffled data
disp(['Boundary match: ' num2str(mean(boundMatch(1,:))) ...
      ' (null:' num2str(mean(boundMatch(2,:))) ')']);
disp(['Log-likelihood: ' num2str(mean(LL(1,:))) ...
      ' (null:' num2str(mean(LL(2,:))) ')']);
   
figure;
subplot(2,1,1);
imagesc(simulData2);
xlabel('Timepoints');
ylabel('Voxels');
subplot(2,1,2);
loggamma = find_events(eventModel.eventPatterns, ...
                       eventModel.eventVar, ...
                       simulData2);
[~, estEvents] = max(loggamma, [], 2);
plot(estEvents);
xlabel('Timepoints');
ylabel('Event label');
end

function eventLabels = generate_event_labels(T, K, lengthStd)
    eventLabels = zeros(1,T);
    startTR = 1;
    for e = 1:(K-1)
        eventLength = round(((T - startTR + 1) / (K - e + 1)) * ...
                             (1 + lengthStd * randn()));
        eventLength = min(max(eventLength, 1), T-startTR+1 - (K-e+1));
        eventLabels(startTR:(startTR + eventLength)) = e;
        startTR = startTR + eventLength;
    end
    eventLabels(startTR:end) = K;
end

function simulData = generate_data(V, T, eventLabels, eventMeans, noiseStd)
    simulData = zeros(V,T);
    for t = 1:T
        simulData(:,t) = eventMeans(:, eventLabels(t)) + ...
                         noiseStd*randn(V,1);
    end

    simulData = zscore(simulData,[],2);
end