function [data,nninall,nnoutall] = generateDataCasino(numSeqs,seqLength,windowLength,A,B)
numStates = length(A);
stateNames      = {'G','P','B'};
annoNames       = {'a','b','c','d','e','f'};
data            = repmat(struct(),1,numSeqs);
nninall       = [];
nnoutall      = [];

for i = 1:numSeqs
    data(i).name        = ['Test' mat2str(i)];
    [o1,s1]             = hmmgenerate(seqLength,A,B);
    data(i).obs         = o1;
    data(i).states      = s1;  
    data(i).seq         = cell2mat(stateNames(s1));
    data(i).annotation  = cell2mat(annoNames(o1));
    [nninput,nnoutput]  = createNNdata(o1,s1,windowLength,numStates);
    data(i).nninput     = nninput;
    data(i).nnoutput    = nnoutput;
    
    nninall             = [nninall; nninput];
    nnoutall            = [nnoutall; nnoutput]; 
end




    function [input,output] = createNNdata(X,y,d,numStates)
        % create sliding window data
        % d is the window width
        assert(mod(d,2) ~= 0, 'Window length must be an odd number');
        numSeqs = length(X);        
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
            for r = 1:N
                row = [];
                diceVal = windows(r,:)+1;   % to fix matlab one indexing
                for j = 1:d
                    thisDice = diceLookup(diceVal(j),:);
                    row = [row,thisDice];
                    
                end
                input(r,:) = row;
            end
            
            %create y
            ylookup = eye(numStates);
            
            output = zeros(N,size(ylookup,2));
            for d =1:N
                output(d,:) = ylookup(y(d),:);
            end
    end
end
