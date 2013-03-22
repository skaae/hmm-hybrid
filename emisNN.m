function B = emisNN(nn,nninput,states,numStates)
% In HMM this should be P(obs | state), but we use a Neural
% Networks which outputs P(State | obs), sort of uses bayes to
% resolve this...

%Get predictions from network
nn.testing = 1;
nn = nnff(nn, nninput, zeros(size(nninput,1), nn.size(end)));
nn.testing = 0;
nnpred = nn.a{end}';


%counts priors
% using p(o|s) = p(s|o)p(o) / p(s)
statePrior = zeros(numStates,1);
L  = length(states);            %number of state

for n = 1:numStates
    statePrior(n) = sum(states == n) / L;
end

B = nnpred ./ repmat(statePrior,1,L);
%B = nnpred ./ repmat(sum(nnpred,1),size(nnpred,1),1);
end