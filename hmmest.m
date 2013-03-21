function [A,pi] = hmmest(stateSeq,numStates)
%%HMMEST computes MLE estimates for A and pi
%
%  statesSeq is a cellArray of state sequences, can be different lengths
%  numSates is the number of unique states in all sequences, must be
%  numbered from 1:numStates

% Calculates A and Pi as:
%   N_jk  = state transitions from j to k           [MLPP eq. 17.11 p. 592]
%   N^1_j = state counts for inital position        [MLPP eq. 17.11 p. 592]
%   A_jk  = N_jk / sum_k (N_jk)                     [MLPP eq. 17.12 p. 593]
%   pi_j  = N^1_j ( sum_k(N^1_j)                    [MLPP eq. 17.12 p. 593]
%
% This file does not estimate the emission matrix (B) because we assume the
% emission probabilities for each timestep are generated with some other
% model
%
%   reference: Machine Learning: A probabilistic Perspective
%             Kevin P. Murphy
%             Chapter 17

states      = 1:numStates;   %assumes that states are numbered from 1:nstates
nSeqs       = size(stateSeq,2);

A_counts    = zeros(numStates,numStates);
stateCount  = zeros(1,numStates);

for n = 1:nSeqs
    
    sSeq = stateSeq{n};
    L    = length(sSeq);
    
    %count inital probs
    for s = states
        stateCount(s)  = sum(sSeq == s); % number of init pos in state s
    end
    
    % count transistions
    for t=1:L-1
        i                = sSeq(t);
        j                = sSeq(t+1);
        A_counts(i,j)    = A_counts(i,j) + 1;
    end
end
pi = (stateCount / sum(stateCount))';
A = A_counts ./ repmat(sum(A_counts,2),1,numStates);

end