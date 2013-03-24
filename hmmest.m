function [A,B,pi,statePrior] = hmmest(data,numStates,numObsTypes)
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
numSeqs     = length(data);
totalCount  = 0;

pi          = zeros(numStates,1);
A           = zeros(numStates,numStates);
B           = zeros(numStates,numObsTypes);
statePrior  = zeros(numStates,1);

for n = 1:numSeqs  
    states                  = data(n).states;
    obs                     = data(n).obs;
    L                       = length(states);
    totalCount              = totalCount + L;
    
    [Acount,Bcount,piCount,stateCount] = counts(obs,states,L,numStates,numObsTypes); 
    A                       = A + Acount;
    B                       = B + Bcount;  
    pi                      = pi + piCount;
    statePrior              = statePrior + stateCount;
end

assert(totalCount       == sum(sum(A))+numSeqs,'Count mismatch A');
assert(totalCount       == sum(sum(B)),        'Count mismatch B');
assert(numSeqs          == sum(pi),            'Count mismatch pi');
assert(sum(statePrior)  == totalCount,         'State prior mismatch'); 

pi = pi     ./ sum(pi);
A = A       ./ repmat(sum(A,2),1,numStates);
B = B       ./ repmat(sum(B,2),1,numObsTypes);
statePrior = statePrior  ./ totalCount;

    function [Acount,Bcount,piCount,stateCount] = counts(obs,states,L,numStates,numObsTypes)
        Acount      = zeros(numStates,numStates);
        piCount     = zeros(numStates,1);
        Bcount      = zeros(numStates,numObsTypes);
        stateCount  = zeros(numStates,1);
        
        
        for s = 1:numStates
            piCount(s)      = sum(states(:,1) == s);
            stateCount(s)   = sum(states == s);
        end
        
        for t=1:L-1
            i                = states(t);
            j                = states(t+1);
            Acount(i,j)      = Acount(i,j) + 1;
        end
        
        for t = 1:L
            j                = obs(t);
            i                = states(t);
            Bcount(i,j)      = Bcount(i,j) + 1;
        end
    end

end