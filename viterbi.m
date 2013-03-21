function res = viterbi(obsCell,tr,pi,b,stateNames)
%VITERBI calculates the most probable state path for a sequence.
% stateSeq = argmax_{z_{t:T}} p(z_{1:T}|obs_{1:T})
% (MLPP eq. 17.68 p. 612)
%
% EXAMPLE
%   transition = [.9 .1;
%              .05 .95;];
%
%   emission   = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6;...
%               1/10, 1/10, 1/10, 1/10, 1/10, 5/10];
%
%   pi         = [0.5; 0.5];
%   B = emission(:,obs);     %  B_ij = P(obs_i | state_j)
%   stateseq = viterbi(obs, transition,pi,B);
%
%
%  reference: Machine Learning: A probabilistic Perspective
%             Kevin P. Murphy
%             Chapter 17
%
%  B could be replaced with other probabilisitc models, e.g a Neural
%  network


%%setup variables  + work in logspace
numStates = size(tr,1);
numSeqs   = length(obsCell);
logTR   = log(tr);
logPI   = log(pi);


res = {};
for n = 1:numSeqs
    logB    = log(b{n});
    obs     = obsCell{n};
    L       = length(obs);
    logP    = zeros(numStates,L);
    backTrack = zeros(numStates,L);
    
    
    logP(:,1) = logPI + logB(:,1);
    stateSeq = zeros(1,L);
    for t = 2:L                  %loop through model
        for state = 1:numStates  %for each state calc. best value given previous state
            bestVal = -inf;
            bestPTR = 0;
            [val,inner] = max(logP(:,t-1) + logTR(:,state));
            if val > bestVal
                bestVal = val;
                bestPTR = inner;
            end
            
            backTrack(state,t)  = bestPTR;
            logP(state,t)       = logB(state,t) + bestVal;
        end
    end
    
    % decide which of the final states is post probable
    [logP, finalState] = max(logP(:,end));
    
    % Backtrack
    stateSeq(L) = finalState;
    for t = fliplr(1:L-1)
        stateSeq(t) = backTrack(stateSeq(t+1),t+1);
    end
    res{n}.stateSeq = stateSeq;
    res{n}.namedSeq = stateNames(stateSeq);
    res{n}.logP     = logP;
    
end

end
