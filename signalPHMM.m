function signalPHMM()
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



%% setup
close all
addpath('../dataanalysis/');

rng('default');rng(0);
temp = load('Simple_NN_nrHidden20_wd1e-05_lr0.1.mat','nn');
nn   = temp.nn; clear temp

%%
numStates       = nn.size(end);
stateNames      = {'S','C','.'};
numObsTypes     = 21;

%files used to estimate hmm parameters
hmmEstFile   = 'Eukar_noTM_HMM.mat';

hmmEstNames  = {'Eukar_noTM_HMM1',...
                'Eukar_noTM_HMM2',...
                'Eukar_noTM_HMM3'};

hmmEvalNames      = {'Eukar_noTM_HMM5'};

nninputFiles      = {'Eukar_noTM_HMM_NNinput5.mat'};

%% estimate A,B and pi and create HMM model
% estimate on Eukar_noTM_HMM_NNinput1.mat,Eukar_noTM_HMM_NNinput2.mat,Eukar_noTM_HMM_NNinput3.mat
[observed_est,hidden_est] = convertDatastructure(hmmEstNames,hmmEstFile);

A_est   = zeros(numStates,numStates);
B_est   = zeros(numStates,numObsTypes);
pi_est  = zeros(numStates,1);
for n = 1:length(hmmEstNames)
    obsSeqs     = observed_est.(hmmEstNames{n});
    stateSeqs   = hidden_est.(hmmEstNames{n});
    [A_temp,B_temp,pi_temp] = hmmest(obsSeqs,stateSeqs,numStates,numObsTypes);
    A_est  = A_est  + A_temp  ./ length(hmmEstNames);
    B_est  = B_est  + B_temp  ./ length(hmmEstNames);
    pi_est = pi_est + pi_temp ./ length(hmmEstNames);
end

%create HMM model
hmm.A           = A_est;
hmm.B           = B_est;
hmm.pi          = pi_est;
hmm.nn          = nn;
hmm.stateNames  = stateNames;
hmm.numStates   = numStates;
hmm.numObsTypes = numObsTypes;

%clean temp variables
clear A_est A_temp B_est B_temp hidden_est i n numObsTypes numStates ...
      obsSeqs observed_est pi_est pi_temp stateSeqs temp
%% Inference in HMM
% Viterbi calculates the most likely path given the observaations
% The forward backward algorithm calculates P(state_t = i|obs_{1:t})

[observed_eval,hidden_eval] = convertDatastructure(hmmEvalNames,hmmEstFile);
temp = load(nninputFiles{1});

nninput_eval     = temp.seq;
obsSeqs_eval     = observed_eval.(hmmEvalNames{1});
stateSeqs_eval   = hidden_eval.(hmmEvalNames{1});

clear temp observed_eval hidden_eval
pathNN           = viterbiNN(hmm,obsSeqs_eval,stateSeqs_eval,nninput_eval);
pathEMIS         = viterbiEMIS(hmm,obsSeqs_eval);

probsNN          = hmmfbNN(hmm,obsSeqs_eval,stateSeqs_eval,nninput_eval);
probsEMIS        = hmmfbEMIS(hmm,obsSeqs_eval);


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


    function [observed,hidden] = convertDatastructure(varNames,filename)
        % convert data structure and estimate A,B,pi. Return the
        for n = 1:length(varNames)
            temp    = load(filename,varNames{n});
            ns = length(temp.(varNames{n}));
            
            % extracts seqs of annotaions an puts it in the correct data structure
            % i.e cell arrays
            for i = 1:ns
                data = temp.(varNames{n});
                observed.(varNames{n})(i)    = {data(i).seq};
                hidden.(varNames{n})(i)  = {data(i).anno};
            end
        end
    
        
    end
end
