function testHMM()
%%TESTHMM shows how to use hmm functions
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
epochs  = 500;
transition = [.9 .1; 0.05 .95;];
emission   = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6;...
              1/10, 1/10, 1/10, 1/10, 1/10, 5/10];

pi              = [0.5; 0.5];
numStates       = 2;
stateNames      = {'F','L'};
d               = 21;                                   %window size for nn
numSeqs         = 3;
seqLength       = 500;
for i = 1:numSeqs
    rng(i);
    [o1,s1]         = hmmgenerate(seqLength,transition,emission);
    obs{i}          = o1;
    states{i}       = s1;
end
%% training network
[nninput,nnoutput,X,y] = createNNdata(obs,states,d); %use slideing window to generate data for NN
nn = trainNN(X,y,epochs);

%% estimate A,B and pi
B_est              = calcB(nn,nninput,states,numStates);
[A_est,pi_est]     = hmmest(states,numStates);



%% Inference in HMM
% Viterbi calculates the most likely path given the observaations
% The forward backward algorithm calculates P(state_t = i|obs_{1:t})
path                      = viterbi(obs, A_est,pi_est,B_est,stateNames);
[forward,backward,decode] = hmmfb(obs, A_est,pi_est,B_est);





%% create simple plots
viterbiErr = zeros(1,numSeqs);
fs = 14;
for c=1:numSeqs
    viterbiErr(c)  = sum(path{c}.stateSeq ~= states{c});
    
    
    figure();
    hold on
    area(states{c}-1,'FaceColor',0.75*ones(1,3),'EdgeColor',ones(1,3));  %true path in grey
    plot(path{c}.stateSeq-1)                                                      %plot viterbi prdictions
    viterbi_title = sprintf('Viterbi prediction Sequnce %d',c);
    title(viterbi_title,'FontSize',16,'fontWeight','bold');
    xlabel('t','FontSize',fs,'fontWeight','bold')
    ylabel('prediction','FontSize',fs,'fontWeight','bold')
    hold off
    
    figure();
    hold on
    area(states{c}-1,'FaceColor',0.75*ones(1,3),'EdgeColor',ones(1,3));
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
        opts.batchsize          = N;
        opts.plot               = 0;
        
        [nn, ~,~]            = nntrain(nn, X, y,opts);
    end
    function [nninput,nnoutput,X,y] = createNNdata(obs,states,d)
        % create sliding window data
        % d is the window width
        assert(mod(d,2) ~= 0, 'Window length must be an odd number');
        numSeqs = length(obs);
        
        nninput     = [];
        nnoutput    = [];
        
        for s = 1:numSeqs
            X = obs{s};
            y = states{s};
            
            [m,N]           = size(X);
            zero_padding    = zeros(m,floor(d/2));
            X_padded        = [zero_padding,X,zero_padding];
            
            %create zero - one encoding for dice 1-6, row 0 is the special case
            % of 0, which maps to zeros(1,6)
            diceLookup      = [zeros(1,6); eye(6)];
            nfeatures       = size(diceLookup,2);    %number of features pr dice value
            
            
            windows = zeros(N,d); %number of windows is N because of zero padding
            for a = 1:N
                windows(a,:) = X_padded(a:a+d-1);
            end
            
            nn_in_size = d*nfeatures;   % size of nn input layer
            input = zeros(N,nn_in_size);
            for i = 1:N
                row = [];
                diceVal = windows(i,:)+1;   % to fix matlab one indexing
                for j = 1:d
                    thisDice = diceLookup(diceVal(j),:);
                    row = [row,thisDice];
                    
                end
                input(i,:) = row;
            end
            
            %create y
            ylookup = [1,0;
                0,1];
            
            output = zeros(N,size(ylookup,2));
            for b =1:N
                output(b,:) = ylookup(y(b),:);
            end
            nnoutput{s} = output;
            nninput{s}  = input;
        end
        
        X = []; y = [];
        for i = 1:length(nninput)
            X = [X; nninput{i}];
            y = [y; nnoutput{i}];
        end
        
    end
end
