function [ probs ] = hmmfbEMIS(hmm,data)
%HMMFBEMIS forward-backward with emission matrix
%   Uses a neural network to predict "emission" probabilities in the hmm.  

numSeqs = length(data);
probs = cell(1,numSeqs);
for i = 1:numSeqs
    
    % check why i need to normalize
    B_emis                    = hmm.B(:,data(i).obs);
    [forward,backward,decode] = hmmfb(hmm.A, hmm.pi, B_emis);
        
    probs{i}.forward     = forward;
    probs{i}.backward    = backward;
    probs{i}.decode      = decode;
end

end