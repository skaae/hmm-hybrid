function testHMM()
%%TESTHMM shows how to use hmm functions
% Shows the use of the functions
% - hmmgenerate (Matlab function)
% - hmmest      (Estimates A and PI)
% - hmmfbNN     (Forward backward algorithm  hmm-nn hybrid)
% - hmmfbEMIS   (Forward backward algorithm  hmm)
% - viterbiNN   (Find most praobable path hmm-nn hybrid)
% - viterbiEMIS (Find most praobable path hmm)
%
% The data is based on the occasionally dishonest casino examples from
% durbin et. al. 1998




close all

%% Generate data
rng('default');rng(0);
% 
% %% settings for hmm model and generate data
epochs          = 50;
seqLength       = 300;   %printing may if seqLength % 60 != 0
transition = [.9 .1; 0.05 .95;];
emission   = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6;...
              1/10, 1/10, 1/10, 1/10, 1/10, 5/10];

numStates       = 2;
stateNames      = {'F','L'};
numObsTypes     = 6;
d               = 21;                                   %window size for nn
numSeqs         = 1;

for i = 1:numSeqs
    rng(i);
    [o1,s1]         = hmmgenerate(seqLength,transition,emission);
    obsSeqs{i}      = o1;
    stateSeqs{i}    = s1;
end

%% Train Neural network
[nninput,nnoutput,X,y]  = createNNdata(obsSeqs,stateSeqs,d); %use slideing window to generate data for NN
nnSeqs                  = createNNSeqs(X,stateSeqs);
nn                      = trainNN(X,y,epochs);



%% estimate A,B and pi and create HMM model
[A_est,B_est,pi_est]          = hmmest(obsSeqs,stateSeqs,numStates,numObsTypes);

%create HMM model
hmm.A           = A_est;
hmm.B           = B_est;
hmm.pi          = pi_est;
hmm.nn          = nn;
hmm.stateNames  = stateNames;
hmm.numStates   = numStates;


%% Inference in HMM
% Viterbi calculates the most likely path given the observaations
% The forward backward algorithm calculates P(state_t = i|obs_{1:t})
pathNN                    = viterbiNN(hmm,obsSeqs,stateSeqs,nnSeqs);
pathEMIS                  = viterbiEMIS(hmm,obsSeqs);

probsNN                   = hmmfbNN(hmm,obsSeqs,stateSeqs,nnSeqs);
probsEMIS                 = hmmfbEMIS(hmm,obsSeqs);

%% create simple plots
viterbiNNErr    = zeros(1,numSeqs);
viterbiEMISErr  = zeros(1,numSeqs);
fs = 14;
lineLength = 60;
for c=1:numSeqs
    
    s = stateSeqs{c};
    rollLabel = num2str(obsSeqs{c});
    rollLabel(rollLabel == ' ') = [];
    
    
    dielabel = repmat('F',size(rollLabel));
    dielabel(s == 2) = 'L';
    fprintf('\n####### Viterbi predcitions######### \n')
    for i=1:lineLength:seqLength
       
        fprintf('Rolls          : %s\n',rollLabel(i:i+lineLength-1));
        fprintf('Die            : %s\n',dielabel(i:i+lineLength-1));
        fprintf('ViterbiNN      : %s\n',cell2mat(pathNN{c}.namedStates(i:i+lineLength-1)));
        fprintf('ViterbiEMIS    : %s\n\n',cell2mat(pathEMIS{c}.namedStates(i:i+lineLength-1)));
        
        
    end
    
    viterbiNNErr(c)   = sum(pathNN{c}.states    ~= stateSeqs{c});
    viterbiEMISErr(c) = sum(pathEMIS{c}.states  ~= stateSeqs{c});
    
    figure();
    hold on
    area(stateSeqs{c}-1,'FaceColor',0.75*ones(1,3),'EdgeColor',ones(1,3));  %true path in grey
    plot(pathNN{c}.states-1)                                                      %plot viterbi prdictions
    viterbi_title = sprintf('ViterbiNN prediction Sequnce %d',c);
    title(viterbi_title,'FontSize',16,'fontWeight','bold');
    xlabel('t','FontSize',fs,'fontWeight','bold')
    ylabel('prediction','FontSize',fs,'fontWeight','bold')
    hold off
    
    figure();
    hold on
    area(stateSeqs{c}-1,'FaceColor',0.75*ones(1,3),'EdgeColor',ones(1,3));
    dec = probsNN{c}.decode;
    plot(dec(1,:))
    decode_title = sprintf('hmmfbNN algorithm Sequnce %d',c);
    title(decode_title,'FontSize',16,'fontWeight','bold')
    ylabel('P(state_{t} = fair | obs_{1:t})','FontSize',fs,'fontWeight','bold');
    xlabel('t','FontSize',fs,'fontWeight','bold');
    hold off
    
     figure();
    hold on
    area(stateSeqs{c}-1,'FaceColor',0.75*ones(1,3),'EdgeColor',ones(1,3));  %true path in grey
    plot(pathEMIS{c}.states-1)                                                      %plot viterbi prdictions
    viterbi_title = sprintf('ViterbiEMIS prediction Sequnce %d',c);
    title(viterbi_title,'FontSize',16,'fontWeight','bold');
    xlabel('t','FontSize',fs,'fontWeight','bold')
    ylabel('prediction','FontSize',fs,'fontWeight','bold')
    hold off
    
    figure();
    hold on
    area(stateSeqs{c}-1,'FaceColor',0.75*ones(1,3),'EdgeColor',ones(1,3));
    dec = probsEMIS{c}.decode;
    plot(dec(1,:))
    decode_title = sprintf('hmmfbEMIS algorithm Sequnce %d',c);
    title(decode_title,'FontSize',16,'fontWeight','bold')
    ylabel('P(state_{t} = fair | obs_{1:t})','FontSize',fs,'fontWeight','bold');
    xlabel('t','FontSize',fs,'fontWeight','bold');
    hold off
end

fprintf('ViterbiNN error    : %s\n',mat2str(viterbiNNErr));
fprintf('ViterbiEMIS error  : %s\n',mat2str(viterbiEMISErr));


    function nnSeqs = createNNSeqs(X,stateSeqs)
        
        numSeqs = length(stateSeqs);
            rowStart = 0
        for i=1:numSeqs
            rowEnd = rowStart+size(stateSeqs{i},2);
            nnSeqs{i} = X(rowStart+1:rowEnd,:);
            rowStart = rowEnd;
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
