function [ path ] = viterbiEMIS(hmm,obsSeqs)
%VITERBIEMIS viterbi with emission matrix
  

numSeqs = length(obsSeqs);
for i = 1:numSeqs
    B_emis                    = hmm.B(:,obsSeqs{i});
    [states,logP]   = viterbi(obsSeqs{i}, hmm.A, hmm.pi, B_emis);
    
    path{i}.states      = states;
    path{i}.logP        = logP;
    path{i}.namedStates = cell2mat(hmm.stateNames(states));
end

end

