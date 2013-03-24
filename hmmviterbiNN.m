function [ path ] = hmmviterbiNN(hmm,data)
%VITERBINN viterbi with hmm-nn hybrid
%   Uses a neural network to predict "emission" probabilities in the hmm.  

numSeqs = length(data);
path = cell(1,numSeqs);

for i = 1:numSeqs  
    B_nn  = hmmemisNN(hmm.nn,       ...  % nn for predicting "emissions"
                data(i).nninput, ...  % input to nn
                hmm.statePrior);      % priors on each state
    [states,logP]   = hmmviterbi(hmm.A, hmm.pi, B_nn);
    
    path{i}.states      = states;
    path{i}.logP        = logP;
    path{i}.namedStates = cell2mat(hmm.stateNames(states));
end

end

