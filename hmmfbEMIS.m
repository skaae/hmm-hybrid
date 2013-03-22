function [ probs ] = hmmfbEMIS(hmm,obsSeqs)
%HMMFBEMIS forward-backward with emission matrix
%   Uses a neural network to predict "emission" probabilities in the hmm.  

numSeqs = length(obsSeqs);
for i = 1:numSeqs
    
    % check why i need to normalize
    B_emis                    = hmm.B(:,obsSeqs{i});
    [forward,backward,decode] = hmmfb(obsSeqs{i}, hmm.A, hmm.pi, B_emis);
        
    probs{i}.forward     = forward;
    probs{i}.backward    = backward;
    probs{i}.decode      = decode;
end

end