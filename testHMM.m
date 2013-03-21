function testHMM()
%TESTHMM shows how to use hmm functions
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
transition = [.9 .1; 0.05 .95;];
emission   = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6;...
              1/10, 1/10, 1/10, 1/10, 1/10, 5/10];

pi         = [0.5; 0.5];
numStates  = 2;
d           = 21;                                       %window size for nn
[obs,states] = hmmgenerate(300,transition,emission);    %generate hmm data


%% training network
[nninput,nnoutput] = createNNdata(obs,states,d); %use slideing window to generate data for NN
nn                 = trainNN(nninput,nnoutput);
%save('nn.mat','nn')

%% estimate A,B and pi
B_est           = calcB(nn,nninput,states);
[A_est,pi_est]  = hmmest({states},numStates);



%% Inference in HMM
% Viterbi calculates the most likely path given the observaations
% The forward backward algorithm calculates P(state_t = i|obs_{1:t})
path                      = viterbi(obs, A_est,pi_est,B_est);
[forward,backward,decode] = hmmfb(obs, A_est,pi_est,B_est);

%% create simple plots
viterbiErr  =  sum(path ~= states);
disp('Viterbi error:')
disp(viterbiErr)

fs = 14;
f1 = figure();
hold on
area(states-1,'FaceColor',0.75*ones(1,3),'EdgeColor',ones(1,3));  %true path in grey
plot(path-1)                                                      %plot viterbi prdictions
title('Viterbi prediction','FontSize',16,'fontWeight','bold'); 
xlabel('t','FontSize',fs,'fontWeight','bold')
ylabel('prediction','FontSize',fs,'fontWeight','bold')
hold off

f2 = figure();
hold on
area(states-1,'FaceColor',0.75*ones(1,3),'EdgeColor',ones(1,3));
plot(decode(1,:))       
title('forward-backward algorithm','FontSize',16,'fontWeight','bold')
ylabel('P(state_{t} = fair | obs_{1:t})','FontSize',fs,'fontWeight','bold'); 
xlabel('t','FontSize',fs,'fontWeight','bold');
hold off



    function B = calcB(nn,nninput,hidden)
        %calculate probability of each state [p(s)]
        ns = length(hidden);   %number of state
        f = sum(hidden == 1) / ns;   %fair state
        b = sum(hidden == 2) / ns;   %bend state
        %run feedforward in network
        nn.testing = 1;
        nn = nnff(nn, nninput, zeros(size(nninput,1), nn.size(end)));
        nn.testing = 0;
        
        nnpred = nn.a{end}';%for hmm terminology this needs to be transposed
        
        % using p(o|s) = p(s|o)p(o) / p(s)
        % for somereason we can disregard p(o) ??
        nnpred(1,:) = nnpred(1,:) ./ f;
        nnpred(2,:) = nnpred(2,:) ./ b;     
        
        %normalize
        B  = nnpred ./ repmat(sum(nnpred,1),size(nnpred,1),1);

        
        
    end
    function  nn = trainNN(nninput,nnoutput)
        %% ex1 vanilla neural net
        [N,inputsize]   = size(nninput);
        [~,outputsize]  = size(nnoutput);
        nn = nnsetup([inputsize 200 outputsize]);
        nn.activation_function = 'sigm';    %  Sigmoid activation function
        nn.output = 'softmax';
        %nn.errfun = @nntest;               %  misclassification
        nn.errfun = @nnmatthew;
        %the default for errfun is nntest, the default for plotfun is updatefigures
        %  This function is applied to train and optionally validation set should be format [er, notUsed] = name(nn, x, y)
        %opts.plotfun                = @nnplotnntest;
        opts.plotfun                 = @nnplotmatthew;
        opts.numepochs =  100;   %  Number of full sweeps through data
        opts.batchsize = N;  %  Take a mean gradient step over this many samples
        opts.plot = 1;
        nn.learningRate = 1e-1;                %  Sigm require a lower learning rate
        nn.weightPenaltyL2 = 1e-5;
        
        [nn, L,loss] = nntrain(nn, nninput, nnoutput,opts);
        
    end
    function [nninput,nnoutput] = createNNdata(X,y,d)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Create lag space matrix
        
        assert(mod(d,2) ~= 0, 'Window length must be an odd number');
        [m,N] = size(X);
        
        zero_padding = zeros(m,floor(d/2));
        X_padded = [zero_padding,X,zero_padding];
        
        %create zero - one encoding for dice 1-6, row 0 is the special case
        % of 0, which maps to zeros(1,6)
        diceLookup = [zeros(1,6); eye(6)];
        
        nfeatures = size(diceLookup,2);    %number of features pr dice value
        
        
        windows = zeros(N,d); %number of windows is N because of zero padding
        for a = 1:N
            windows(a,:) = X_padded(a:a+d-1);
        end
        
        nn_in_size = d*nfeatures;   % size of nn input layer
        nninput = zeros(N,nn_in_size);
        for i = 1:N
            row = [];
            diceVal = windows(i,:)+1;   % to fix matlab one indexing
            for j = 1:d
                thisDice = diceLookup(diceVal(j),:);
                row = [row,thisDice];
                
            end
            nninput(i,:) = row;
        end
        
        %create y
        ylookup = [1,0;
            0,1];
        
        nnoutput = zeros(N,size(ylookup,2));
        for b =1:N
            nnoutput(b,:) = ylookup(y(b),:);
        end
        
        
    end

 function x =  normalize(x)
    
    z = sum(x(:));
    z(z==0) = 1;
    x = x./z;
    
    end
end
