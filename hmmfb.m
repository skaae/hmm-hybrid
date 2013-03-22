function [ forward,backward,decode] = hmmfb(obs,tr,pi,b)
%HMMFB hmm forward - backward algorithm
% implements the forward backward algorithm. 
% The forward algorithm calculates the probabilites of having generated the
% observation sequence up untill time t and ending in state s.
% 
% The backward algorithm calculates the probability of generating the the
% observation sequence given that the state sequence emerges from state s
% at time t. 
%
% The forward backward algorithm calculates P(state_t = i|obs_{1:t})
%
%   reference: Machine Learning: A probabilistic Perspective
%             Kevin P. Murphy
%             Chapter 17
%   or 
%            : Non-linear signalprocessing lecture notes ex 12

%% EQUATIONS
% FORWARD: $$\alpha(y_1^{t},i) = \sum_{j=1}^S \alpha(y_1^{t-1},j) a(i|j)b(y_{t}|i) $$
% 
% $$  \alpha(y_1^1,i) = P(x_1=i)b(y_1|i) $$
% 
% BACKWARD : $$\beta(y_{t+1}^T|i) = \sum_{j=1}^S \beta(y_{t+2}^T|j) a(j|i)b(y_{t+1}|j) $$
% 
% $$  \beta(y_{T+1}^T|i) = 1 $$


%% setup variables  + work in logspace
numStates = length(tr);
L         = length(obs);
forward   = zeros(numStates,L);   %vector for storing probabilities


%% FORWARD ALGORITHM
forward(:,1) = pi .* b(:,1);
for t = 2:L                     %loop through model
    for state = 1:numStates     %for each state calc. best value given previous sequence probs
        val                 = sum(forward(:,t-1).*tr(:,state));
        forward(state,t)    =  b(state,t) .* val; 
    end
    forward(:,t) =  normalize(forward(:,t));    %normalize
end

%% BACKWARD ALGORITHM
backward        = zeros(numStates,L); 
backward(:,L)   = ones(numStates,1);
for t = fliplr(1:L-1)
        backward(:,t)   = normalize(tr * (backward(:,t+1) .* b(:,t+1)));
end

%% DECODE 
temp = forward .* backward;
decode = temp ./ repmat(sum(temp,1),numStates,1);

    function x =  normalize(x)
    
    z = sum(x(:));
    z(z==0) = 1;
    x = x./z;
    
    end
end
