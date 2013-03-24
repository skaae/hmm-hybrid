function [ probs ] = hmmfbNN(hmm,data)
%HMMFBNN forward-backward with hmm-nn hybrid
%   Uses a neural network to predict "emission" probabilities in the hmm.  

numSeqs = length(data);
probs = cell(1,numSeqs);
for i = 1:numSeqs
    
    
    B_nn  = hmmemisNN(hmm.nn,       ...  % nn for predicting "emissions"
                data(i).nninput, ...  % input to nn
                hmm.statePrior);
    
    
    [forward,backward,decode]= hmmfb(data(i).obs, hmm.A, hmm.pi, B_nn);
        
    probs{i}.forward     = forward;
    probs{i}.backward    = backward;
    probs{i}.decode      = decode;
end

end