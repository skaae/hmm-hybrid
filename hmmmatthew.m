function corr = hmmmatthew(path, data,numStates)
%HMMconfmat calculates the confusion matrix
for i = 1:length(path)  % iterate over states
    
    mat = zeros(2,2,numStates);
    for target_class = 1:numStates    % testing: set to four
        %create binary vectors for each class. For each class (target_class)
        % match the predition with target class and the expected class with the
        % target class
        pred     = path{i}.states;
        expected = data(i).states;
        
        
        pred_class = ~(pred     == target_class);
        true_class = ~(expected == target_class);
        
        %
        [TP,TN,FP,FN] =  confusion(pred_class,true_class);
        
        mat(:,:,target_class) = mat(:,:,target_class) + [TP, FN; FP, TN];
    end
    
end

corr = repmat(struct(),1,numStates);
for i=1:numStates
    corr(i).confusion = mat(:,:,i);
    TP = corr(i).confusion(1,1);
    FN = corr(i).confusion(1,2);
    FP = corr(i).confusion(2,1);
    TN = corr(i).confusion(2,2);
    corr(i).mcc = (TP * TN - FP * FN) ./ sqrt( (TP+FP) * (TP+FN) * (TN+FP) * (TN + FN) );
end

    function [TP,TN,FP,FN]=  confusion(pred_class,true_class)
        TP = sum( (pred_class == true_class) .* (true_class == 0) ); %True positive
        TN = sum( (pred_class == true_class) .* (true_class == 1) ); %True negative
        FP = sum( (pred_class ~= true_class) .* (pred_class == 1) ); %False positive
        FN = sum( (pred_class ~= true_class) .* (pred_class == 0) ); %False negative
        
    end
end

