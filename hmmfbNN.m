function [ probs ] = hmmfbNN(hmm,obsSeqs,stateSeqs,nnSeqs)
%HMMFBNN forward-backward with hmm-nn hybrid
%   Uses a neural network to predict "emission" probabilities in the hmm.  

numSeqs = length(obsSeqs);
for i = 1:numSeqs
    
    
    B_nn = emisNN(hmm.nn, nnSeqs{i}, stateSeqs{i}, hmm.numStates);
    [forward,backward,decode]= hmmfb(obsSeqs{i}, hmm.A, hmm.pi, B_nn);
        
    probs{i}.forward     = forward;
    probs{i}.backward    = backward;
    probs{i}.decode      = decode;
end

end