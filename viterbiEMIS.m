function [ path ] = viterbiEMIS(hmm,data)
%VITERBIEMIS viterbi with emission matrix
  

numSeqs = length(data);
path = cell(1,numSeqs);
for i = 1:numSeqs
    B_emis          = hmm.B(:,data(i).obs);
    [states,logP]   = viterbi(data(i).obs, hmm.A, hmm.pi, B_emis);
    
    path{i}.states      = states;
    path{i}.logP        = logP;
    path{i}.namedStates = cell2mat(hmm.stateNames(states));
end

end

