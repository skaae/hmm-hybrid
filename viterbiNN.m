function [ path ] = viterbiNN(hmm,obsSeqs,stateSeqs,nnSeqs)
%VITERBINN viterbi with hmm-nn hybrid
%   Uses a neural network to predict "emission" probabilities in the hmm.  

numSeqs = length(obsSeqs);
for i = 1:numSeqs
    
    
    B_nn            = emisNN(hmm.nn, nnSeqs{i}, stateSeqs{i}, hmm.numStates);
    [states,logP]   = viterbi(obsSeqs{i}, hmm.A, hmm.pi, B_nn);
    
    path{i}.states      = states;
    path{i}.logP        = logP;
    path{i}.namedStates = hmm.stateNames(states);
end

end

