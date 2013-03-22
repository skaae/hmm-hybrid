% scritp for creating data to use in HMM. The final data is cell array
% where seq{i} correspond to the input needed for for a neural
% network to produce B in the HMM. I.e if AMINOACIDSEQUENCE i is n long,
% and the NN takes m features, then seq{i} would be a n by m matrix.
addpath('../dataanalysis');

numSplits = 5;
input       = {'Eukar_input1','Eukar_input2','Eukar_input3',...
                'Eukar_input4','Eukar_input5'};

seqsHMM     = {'Eukar_noTM_HMM1','Eukar_noTM_HMM2','Eukar_noTM_HMM3',...
                'Eukar_noTM_HMM4','Eukar_noTM_HMM5'};

outFiles    = {'Eukar_noTM_HMM_NNinput1','Eukar_noTM_HMM_NNinput2',...
                'Eukar_noTM_HMM_NNinput3','Eukar_noTM_HMM_NNinput4',...
                'Eukar_noTM_HMM_NNinput5'};

for i = 1:numSplits
temp = load('all_window_size41.mat',input{i});
X = temp.(input{i});

temp = load('Eukar_noTM_HMM.mat',seqsHMM{i});
seqs = temp.(seqsHMM{i});
clear temp
 
numSeqs = size(seqs,2);
rowStart = 0;

for n = 1:numSeqs
    rowEnd = rowStart+size(seqs(n).seq,2);
    seq{n} = X(rowStart+1:rowEnd,:);
    rowStart = rowEnd;
end
save(['../dataanalysis/',outFiles{i}],'seq');

clear seq

end
clear all
