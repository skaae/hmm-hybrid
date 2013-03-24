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
epochs          = 30;
seqLength       = 120;   %printing may if seqLength % 60 != 0
transition = [.95 .025 .025;...
    .01 .95 .04;
    .04 .01 .95];
emission   = [1/60, 1/60, 1/60, 1/60, 1/60, 1/60;...
    1/10, 1/10, 1/10, 0/10, 0/10, 7/10;...
    7/10, 0/10, 0/10, 1/10, 1/10, 1/10];

numStates       = 3;
stateNames      = {'G','P','B'};
numObsTypes     = 6;
d               = 11;                                   %window size for nn
numSeqs         = 25;

numSeqs_eval    = 10;

for i = 1:numSeqs
    rng(i);
    [o1,s1]         = hmmgenerate(seqLength,transition,emission);
    obsSeqs{i}      = o1;
    stateSeqs{i}    = s1;
end


rng(10);
[o1,s1]         = hmmgenerate(seqLength,transition,emission);
obsSeqs_val{i}    = o1;
obsSeqs_val{i}    = s1;


%% Train Neural network
[~,~,X_train,y_train]  = createNNdata(obsSeqs,stateSeqs,d,numStates); %use slideing window to generate data for NN
[~,~,X_val,y_val]  = createNNdata(obsSeqs,stateSeqs,d,numStates); %use slideing window to generate data for NN

nn                      = trainNN(X_train,y_train,X_val,y_val,epochs);



%% estimate A,B and pi and create HMM model
[A_est,B_est,pi_est]          = hmmest(obsSeqs,stateSeqs,numStates,numObsTypes);

%create HMM model
hmm.A           = A_est;
hmm.B           = B_est;
hmm.pi          = pi_est;
hmm.nn          = nn;
hmm.stateNames  = stateNames;
hmm.numStates   = numStates;

clear A_est B_est X_train X_val i nnSeqs nninput nnoutput o1 obsSeqs pi_est s s1 ...
    stateSeqs y_train y_val epochs nn


%generate test data
rng(33);
[o1,s1]              = hmmgenerate(seqLength,transition,emission);
obsSeqs_eval{1}      = o1;
stateSeqs_eval{1}    = s1;
[nninput_eval,nnoutput_eval,X_eval,y_eval]= createNNdata(...
    obsSeqs_eval, stateSeqs_eval,d,numStates);
nnSeqs_eval          = createNNSeqs(X_eval,stateSeqs_eval);

clear d clear emission i o1 transition emission s1 X_eval y_eval


%% Inference in HMM
% Viterbi calculates the most likely path given the observaations
% The forward backward algorithm calculates P(state_t = i|obs_{1:t})
pathHMMNN    = viterbiNN(hmm,obsSeqs_eval,stateSeqs_eval,nnSeqs_eval);
pathHMM      = viterbiEMIS(hmm,obsSeqs_eval);
pathNN       = nnpredict(hmm.nn,nninput_eval{1})';
probsHMMNN   = hmmfbNN(hmm,obsSeqs_eval,stateSeqs_eval,nnSeqs_eval);
probsHMM     = hmmfbEMIS(hmm,obsSeqs_eval);

%% create simple plots
fs = 14;
lineLength = 60;

color = [179, 226, 205;
    253, 205, 172;
    203, 213, 232] ./ 255;

color2 = [27, 158, 119; 217, 95, 2; 117, 112, 179;]./255;
for c=1:numSeqs
    o                = mat2str(obsSeqs_eval{c});
    o = strrep(o,' ',''); o = strrep(o,'[','');
    obs = strrep(o,']','');


    
    true_stat_names     = cell2mat(stateNames(stateSeqs_eval{c}));
    pstat_HMMNN_names   = pathHMMNN{c}.namedStates;
    pstat_HMM_names     = pathHMM{c}.namedStates;
    pstat_NN_names      = cell2mat(stateNames(pathNN));
    

    
    fprintf('\n####### Viterbi predcitions######### \n')
    
    fprintf('Viterbi HMM-NN error    : %d\n',sum(pathHMMNN{c}.states    ~= stateSeqs_eval{c}));
    fprintf('Viterbi HMM error       : %d\n',sum(pathHMM{c}.states  ~= stateSeqs_eval{c}));
    fprintf('Viterbi NN error        : %d\n\n',sum(pathNN  ~= stateSeqs_eval{c}));
    for i=1:lineLength:seqLength
        
        fprintf('Rolls              : %s\n'   ,obs(i:i+lineLength-1));
        fprintf('Die                : %s\n'   ,true_stat_names(i:i+lineLength-1));
        fprintf('Viterbi (HMM-NN)   : %s\n'   ,pstat_HMMNN_names(i:i+lineLength-1));
        fprintf('ViterbiEMIS (HMM)  : %s\n' ,pstat_HMM_names(i:i+lineLength-1));
        fprintf('ViterbiEMIS (NN)   : %s\n\n' ,pstat_NN_names(i:i+lineLength-1));
        
        
    end
    figure();
    hold on
    backgroundWidth = 1;
    for q=1:numStates
        bar(1:seqLength,(stateSeqs_eval{c}==q)*numStates,backgroundWidth,'FaceColor',color(:,q),'EdgeColor','none')
    end
    plot(pathHMMNN{c}.states)                                                      %plot viterbi prdictions
    viterbi_title = sprintf('ViterbiNN prediction Sequnce %d',c);
    title(viterbi_title,'FontSize',16,'fontWeight','bold');
    xlabel('t','FontSize',fs,'fontWeight','bold')
    ylabel('prediction','FontSize',fs,'fontWeight','bold')
    
    set(gca,'XLim',[0 seqLength])
    set(gca,'YTick',1:numStates)
    set(gca,'YTickLabel',stateNames)
    set(gca,'YLim',[0,numStates])
    
    hold off
    
    
    figure();
    hold on
    
    for q=1:numStates
        bar(1:seqLength,stateSeqs_eval{c}==q,backgroundWidth,'FaceColor',color(q,:),'EdgeColor','none')
    end
    
    dec = probsHMMNN{c}.decode;
    for q=1:numStates
        plot(dec(q,:),'color',color2(q,:));
    end
    decode_title = sprintf('hmmfbNN algorithm Sequnce %d',c);
    title(decode_title,'FontSize',16,'fontWeight','bold')
    ylabel('P(state_{t} = fair | obs_{1:t})','FontSize',fs,'fontWeight','bold');
    xlabel('t','FontSize',fs,'fontWeight','bold');
    set(gca,'XLim',[0 seqLength])
    
    set(gca,'YLim',[0,1])
    hold off
end



    function nnSeqs = createNNSeqs(X,stateSeqs)
        
        numSeqs = length(stateSeqs);
        rowStart = 0;
        for i=1:numSeqs
            rowEnd = rowStart+size(stateSeqs{i},2);
            nnSeqs{i} = X(rowStart+1:rowEnd,:);
            rowStart = rowEnd;
        end
    end

    function  nn = trainNN(X_train,y_train,X_val,y_val,epochs)
        % Trains a neural netowork from sliding window data
        [N,inputsize]           = size(X_train);
        [~,outputsize]          = size(y_train);
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
        
        [nn, ~,~]            = nntrain(nn, X_train, y_train,opts, X_val, y_val);
    end

    function [nninput,nnoutput,X,y] = createNNdata(obs,states,d,numStates)
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
            ylookup = eye(numStates);
            
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