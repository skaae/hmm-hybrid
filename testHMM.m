function testHMM()
%%TESTHMM shows how to use hmm functions
% Shows the use of the functions
% - hmmgenerate (Matlab function)
% - hmmest      (Estimates A and PI)
% - hmmfbNN     (Forward backward algorithm  hmm-nn hybrid)
% - hmmfbEMIS   (Forward backward algorithm  hmm)
% - viterbiNN   (Find most praobable path hmm-nn hybrid)
% - viterbiEMIS (Find most praobable path hmm)
% - hmmplot
% The data is based on the occasionally dishonest casino examples from
% durbin et. al. 1998

close all

%% Generate data
rng('default');rng(0);
%
% %% settings for hmm model and generate data
doPlots = 1;   %if probs should be plotted in figures


epochs          = 30;
seqLength       = 210;   %printing may if seqLength % 70 != 0
transition = [.95 .025 .025;...
    .01 .95 .04;
    .04 .01 .95];
emission   = [1/60, 1/60, 1/60, 1/60, 1/60, 1/60;...
    1/10, 1/10, 1/10, 0/10, 0/10, 7/10;...
    7/10, 0/10, 0/10, 1/10, 1/10, 1/10];

numStates       = 3;
stateNames      = {'G','P','B'};
obsNames        = {'1','2','3','4','5','6'};
numObsTypes     = 6;
windowLength    = 11;                                

numPlots        = [1,3];   %sequences to plot i.e plots seq 1 and 3
numSeqs_train   = 25;
numSeqs_val     = 10;
numSeqs_test    = 5;

%% generate training, validation and test data
[data_train,nnin_train,nnout_train] = hmmgenerateDataCasino(...
    numSeqs_train,seqLength,windowLength,transition,emission);

[~,nnin_val,nnout_val] = hmmgenerateDataCasino(...
    numSeqs_val,seqLength,windowLength,transition,emission);

[data_test,~,~] = hmmgenerateDataCasino(...
    numSeqs_test,seqLength,windowLength,transition,emission);


%% Train Neural network
[N,inputsize]           = size(nnin_train);
[~,outputsize]          = size(nnout_train);
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

[nn, ~,~]            = nntrain(nn, nnin_train, nnout_train,opts, nnin_val, nnout_val);

clear nnin_train nnout_train nnin_val nnout_val N inputsize outputsize epochs ...

%% estimate A,B and pi and create HMM model
[A_est,B_est,pi_est,statePrior]    = hmmest(data_train,numStates,numObsTypes);

%create HMM model
hmm.A                   = A_est;
hmm.B                   = B_est;
hmm.pi                  = pi_est;
hmm.statePrior          = statePrior;
hmm.nn                  = nn;
hmm.stateNames          = stateNames;
hmm.numStates           = numStates;

clear A_est B_est pi_est nn data_train statePrior


%generate test data
%% Inference in HMM
% Viterbi calculates the most likely path given the observaations
% The forward backward algorithm calculates P(state_t = i|obs_{1:t})
pathHMMNN    = hmmviterbiNN(hmm,data_test);
pathHMM      = hmmviterbiEMIS(hmm,data_test);

pathNN = cell(1,numSeqs_test);
for i = 1:numSeqs_test
    pathNN{i}.states       = nnpredict(hmm.nn,data_test(i).nninput)';
end
probsHMMNN   = hmmfbNN(hmm,data_test);
probsHMM     = hmmfbEMIS(hmm,data_test);


%% calculate matthew correlation
matthewHMMNN = hmmmatthew(pathHMMNN, data_test,numStates);
matthewHMM   = hmmmatthew(pathHMM, data_test,numStates);
matthewNN    = hmmmatthew(pathNN, data_test,numStates);

fprintf('\n  MATTHEW CORRELATIONS \n');
classstr ='class      ';
mHMMNN  = 'HMMNN    : '; 
mHMM    = 'HMM      : '; 
mNN     = 'NN       : ';
f = '%1.3f';
for i = 1:numStates
   mHMMNN   =  [mHMMNN num2str(matthewHMMNN(i).mcc,f) '\t'];
   mHMM     =  [mHMM num2str(matthewHMM(i).mcc,f) '\t'];
   mNN      =  [mNN num2str(matthewNN(i).mcc,f) '\t'];
   classstr =  [classstr,['   ',stateNames{i},'   ' '\t']];
end

fprintf([classstr '\n' ...
         mHMMNN   '\n' ...
         mHMM     '\n' ...
         mNN      '\n']);
%% viterbi errors
    vHMMNN = 0; vHMM = 0; vNN = 0;
    for c = 1:numSeqs_test
        vHMMNN = vHMMNN + sum(pathHMMNN{c}.states ~= data_test(c).states);
        vHMM   = vHMM   + sum(pathHMM{c}.states   ~= data_test(c).states);
        vNN    = vNN    + sum(pathNN{c}.states     ~= data_test(c).states);
    end
    
    fprintf('\n     VITERBI PREDICTIONS\n');
    fprintf('Viterbi HMM-NN error    : %d\n',vHMMNN);
    fprintf('Viterbi HMM error       : %d\n',vHMM);
    fprintf('Viterbi NN error        : %d\n\n',vNN);


%% create simple plots
colorBck = [179, 226, 205;
    253, 205, 172;
    203, 213, 232] ./ 255;

colorLines = [27, 158, 119; 217, 95, 2; 117, 112, 179;]./255;
hmmplotHMMNN(pathHMMNN, pathHMM, pathNN, probsHMMNN,... %infered paths
          data_test, obsNames ,stateNames,numStates,numPlots,colorBck, colorLines,doPlots)
end