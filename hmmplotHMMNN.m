function hmmplotHMMNN(pathHMMNN, pathHMM, pathNN, probsHMMNN, data,stateNames,numStates,colorBck, colorLines)
%hmmplotHMMNN Summary of this function goes here
%   Detailed explanation goes here
numSeqs = length(data);
seqLength = length(data(1).obs);

fs = 14;
lineLength = 60;
legendPrefix = 'True state: ';


for c=1:numSeqs
    fprintf('\n\n------------------SEQUENCE %i-----------------------------\n',c);
    obs                 = vect2str(data(c).obs,'formatstring','%i', ...
          'openingDelimiter','','closingDelimiter','','separator','');
    true_stat_names     = cell2mat(stateNames(data(c).states));
    pstat_HMMNN_names   = pathHMMNN{c}.namedStates;
    pstat_HMM_names     = pathHMM{c}.namedStates;
    pstat_NN_names      = cell2mat(stateNames(pathNN{c}.states));
    

    for i=1:lineLength:seqLength
        
        fprintf('observation        : %s\n'   ,obs(i:i+lineLength-1));
        fprintf('true state         : %s\n'   ,true_stat_names(i:i+lineLength-1));
        fprintf('Viterbi (HMM-NN)   : %s\n'   ,pstat_HMMNN_names(i:i+lineLength-1));
        fprintf('Viterbi (HMM)      : %s\n' ,pstat_HMM_names(i:i+lineLength-1));
        fprintf('Viterbi (NN)       : %s\n\n' ,pstat_NN_names(i:i+lineLength-1));
        
        
    end
    figure();
    hold on
    backgroundWidth = 1;
    l = [];
    for q=1:numStates
        l(end+1)=bar(1:seqLength,(data(c).states==q)*numStates,...
            backgroundWidth, 'FaceColor',colorBck(:,q),'EdgeColor','none');
    end
    l(end+1)=plot(pathHMMNN{c}.states);                                                      %plot viterbi prdictions
    viterbi_title = sprintf('ViterbiNN prediction Sequnce %d',c);
    title(viterbi_title,'FontSize',16,'fontWeight','bold');
    xlabel('t','FontSize',fs,'fontWeight','bold')
    ylabel('prediction','FontSize',fs,'fontWeight','bold')
    
    set(gca,'XLim',[0 seqLength])
    set(gca,'YTick',1:numStates)
    set(gca,'YTickLabel',stateNames)
    set(gca,'YLim',[0,numStates])
    
    legendNames = cell(1,numStates+1);
    legendPrefix = 'True State: ';
    for i = 1:numStates
        st = [legendPrefix,stateNames{i}];
        legendNames{1,i} = st;
    end
    legendNames{1,numStates+1} = 'Viterbi Pred.';
    legend(l,legendNames,'Location','Best');
    hold off
    
    
    figure();
    hold on
    
    l2 = [];
    
    for q=1:numStates
        l2(end+1) = bar(1:seqLength,data(c).states==q,backgroundWidth,...
            'FaceColor',colorBck(q,:),'EdgeColor','none');
    end    
    dec = probsHMMNN{c}.decode;
    for q=1:numStates
        l2(end+1) = plot(dec(q,:),'color',colorLines(q,:));
    end
    decode_title = sprintf('hmmfbNN algorithm Sequnce %d',c);
    title(decode_title,'FontSize',16,'fontWeight','bold')
    ylabel('P(state_{t} = fair | obs_{1:t})','FontSize',fs,'fontWeight','bold');
    xlabel('t','FontSize',fs,'fontWeight','bold');
    set(gca,'XLim',[0 seqLength])
    set(gca,'YLim',[0,1])
    legendNames = cell(1,2*numStates);
    legendPrefix = 'True State: ';
    for i = 1:numStates
        st = [legendPrefix,stateNames{i}];
        legendNames{1,i} = st;
    end
    
    legendPrefix = 'hmmfb: ';
    for i = 1:numStates
        st = [legendPrefix,stateNames{i}];
        legendNames{1,numStates+i} = st;
    end
    legend(l2,legendNames,'Location','Best');
    hold off
end

end

