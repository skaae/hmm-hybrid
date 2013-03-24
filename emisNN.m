function B = emisNN(nn,nninput,statePrior)
% In HMM this should be P(obs | state), but we use a Neural
% Networks which outputs P(State | obs), sort of uses bayes to
% resolve this...

%Get predictions from network
nn.testing = 1;
nn = nnff(nn, nninput, zeros(size(nninput,1), nn.size(end)));
nn.testing = 0;
nnpred = nn.a{end}';

L = size(nnpred,2);
B = nnpred ./ repmat(statePrior,1,L);
%B = nnpred ./ repmat(sum(nnpred,1),size(nnpred,1),1);
end