function s = logsumexp(x)
%LOGSUMEXP computes the sum of log probabilities numerically stable 
% This computes log(sum(exp(a), 1)) in a numerically stable way
  maxs_small = max(x, [], 1);
  maxs_big = repmat(maxs_small, [size(x, 1), 1]);
  s = log(sum(exp(x - maxs_big), 1)) + maxs_small;
end
