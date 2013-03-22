function signalPHMM()
addpath ../dataanalysis
%%signalPHMM shows how to use hmm functions
% Shows the use of the functions
% - hmmgenerate (Matlab function)
% - hmmest   (Estimates A and PI)
% - hmmfb    (Forward backward algorithm)
% - viterby  (Find most praobable path)
%
% The data is based on the occasionally dishonest casino examples from
% durbin et. al. 1998

%% setings
close all
rng('default');rng(0);

%% settings for hmm model and generate data

temp = load('all_window_size41_small.mat');
X = {temp.Eukar_input1};
y = {temp.Eukar_target1};
numSeqs = size(X,1);
windowSize = 21
epochs  = 1;
numStates       = 4;
stateNames      = {'S','C','T','.'};

[states,stateNames,obs] = convToVec(y{1},X{1},numStates,stateNames,windowSize);



%%



%% training network
nn = trainNN(X{1},y{1},epochs);


%% estimate A,B and pi
B_est              = calcB(nn,X,{states},numStates);
[A_est,pi_est]     = hmmest({states},numStates);



%% Inference in HMM
% Viterbi calculates the most likely path given the observaations
% The forward backward algorithm calculates P(state_t = i|obs_{1:t})
path                      = viterbi({X}, A_est,pi_est,B_est,stateNames);
[forward,backward,decode] = hmmfb({X}, A_est,pi_est,B_est);





%% create simple plots
viterbiErr = zeros(1,numSeqs);
fs = 14;
for c=1:numSeqs
    viterbiErr(c)  = sum(path{c}.stateSeq ~= states);
    
    
    figure();
    hold on
    %area(states-1,'FaceColor',0.75*ones(1,3),'EdgeColor',ones(1,3));  %true path in grey
    plot(path{c}.stateSeq-1)                                                      %plot viterbi prdictions
    viterbi_title = sprintf('Viterbi prediction Sequnce %d',c);
    title(viterbi_title,'FontSize',16,'fontWeight','bold');
    xlabel('t','FontSize',fs,'fontWeight','bold')
    ylabel('prediction','FontSize',fs,'fontWeight','bold')
    hold off
    
    figure();
    hold on
    %area(states-1,'FaceColor',0.75*ones(1,3),'EdgeColor',ones(1,3));
    dec = decode{c};
    plot(dec(1,:))
    decode_title = sprintf('forward-backward algorithm Sequnce %d',c);
    title(decode_title,'FontSize',16,'fontWeight','bold')
    ylabel('P(state_{t} = fair | obs_{1:t})','FontSize',fs,'fontWeight','bold');
    xlabel('t','FontSize',fs,'fontWeight','bold');
    hold off
end
disp('Viterbi error:')
disp(viterbiErr)

    function [stateSeq,nameSeq,obs] = convToVec(y,X,numStates,stateNames,windowSize)
        [m,n] = size(y);
        [mx,nx] = size(X);
        stateSeq = zeros(m,1);    
        convert = eye(numStates);
        for i=1:numStates
          v = repmat(convert(i,:),m,1);
          stateSeq = stateSeq +  all(y == v,2).* i;
        end
        nameSeq = stateNames(stateSeq);
        
        inputsPrAA = nx / windowSize;
        pattern = [ones(1,21), zeros(1,inputsPrAA - 21)];
        mask = logical(repmat(pattern,1,windowSize));
        X_masked = X(:,mask);
        
        
        obs = zeros(m,1);    
        convert = eye(21);
        
            for i=1:21
                for j = 1:21:size(X_masked,2) 
                v = repmat(convert(i,:),m,1);
                obs = obs +  all(X_masked(:,j:j+21-1) == v,2).* i;
            end
        end
        
        
    end


    function B = calcB(nn,nninput,statesCell,numStates)
        % In HMM this should be P(obs | state), but we use a Neural
        % Networks which outputs P(State | obs), sort of uses bayes to
        % resolve this...
        
        %Get predictions from network
        numSeqs = length(nninput);
        
        for s = 1:numSeqs
            data = nninput{s};
            nn.testing = 1;
            nn = nnff(nn, data, zeros(size(data,1), nn.size(end)));
            nn.testing = 0;
            nnpred{s} = nn.a{end}';
        end
        
        %counts priors
        % using p(o|s) = p(s|o)p(o) / p(s)
        % for somereason we can disregard p(o) ??
        
        statePrior = zeros(1,numStates);
        L = 0;
        for i = 1:numSeqs
            stateSeq = statesCell{i};
            L  = L+length(stateSeq);            %number of state
            
            for n = 1:numStates
                statePrior(n) = sum(stateSeq == n);
            end
            for n=1:numStates
                statePrior(n) = statePrior(n) ./ L;
                np = nnpred{i};
                np(n,:) = np(n,:) ./ statePrior(n);
                
            end
            B{i} = np ./ repmat(sum(np,1),size(np,1),1);
        end
        
        
        
    end

    function  nn = trainNN(X,y,epochs)
        % Trains a neural netowork from sliding window data
        [N,inputsize]           = size(X);
        [~,outputsize]          = size(y);
        nn                      = nnsetup([inputsize 100 outputsize]);
        nn.activation_function  = 'tanh_opt';
        nn.normalize            = 1;
        nn.output               = 'softmax';
        nn.errfun               = @nnmatthew;
        nn.learningRate         = 1e-1;
        nn.weightPenaltyL2      = 1e-5;
        
        opts.plotfun            = @nnplotmatthew;
        opts.numepochs          = epochs;
        opts.batchsize          = 500;
        opts.plot               = 0;
        
        [nn, ~,~]            = nntrain(nn, X, y,opts);
    end
end
