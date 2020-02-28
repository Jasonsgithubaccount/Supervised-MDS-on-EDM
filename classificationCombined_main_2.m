function [Accuracy, data_train_ori, target_train_ori, data_test_ori, target_test_ori] = classificationCombined_main_2(filename_train, varargin)
%  This code is to combine all possible situation.
%
%  INPUTS:
%
%       filename_train£º the name of training data file. This code is
%       dicided to 
%
%       dim:
%
%       alpha:
%
%       q:
%
%       disttype:
%
%
%% 

% Open a new file 'Output.txt' for recording.
fclose('all'); 
if exist('Output_2.txt') ==2; delete('Output_2.txt'); end
fid = fopen('Output_2.txt','a+');fclose(fid); 

% Identify the extension of the input training data file 
i = find(filename_train=='.');
if ~isempty(i) 
    l = length(filename_train);
    l_i = length(i);
    ex_train = filename_train(i(l_i)+1:l);
else 
    ex_train = '';
end
    

% Construct training data matrix
switch ex_train
    
    case 'dm'
        
        [data_train_ori, target_train_ori] = dm2data(filename_train);
        
    case 'tsv'
        
        [data_train_ori, target_train_ori] = tsv2data(filename_train);
        
    case ''
        
        [data_train_ori, target_train_ori] = simul_data_const(filename_train);
        
end
        
% Construct testing data matrix   
if strcmp(char(varargin(1)),'test')

    filename_test = char(varargin(2));
    i = find(filename_test=='.');
    if ~isempty(i) 
        l = length(filename_test);
        l_i = length(i);
        ex_test = filename_test(i(l_i)+1:l);
        if ~strcmp(ex_test, ex_train)
            error('This file does not match the training file.');
        else
            if ~strcmp(ex_test, 'tsv')
                error('This type of file cannot be treated as testing file.');
            end
        end
        
    else 
        error('Name of testing file should be with extension.');
    end
   [data_test_ori, target_test_ori] = tsv2data(filename_test);

end
    
    

switch char(varargin(1))
    
    case 'k-fold'
        
       k = str2num(cell2mat(varargin(2)));
        
    case 'test'
        
        % optimized discriminant model by matlab
        
        Mdl_Disc = fitcecoc(data_train_ori,target_train_ori,...
            'Learners','discriminant',...
            'OptimizeHyperparameters','all',...
            'HyperparameterOptimizationOptions',...
            struct('AcquisitionFunctionName', 'expected-improvement-plus'));
        target_hat = predict(Mdl_Disc, data_test_ori);
        Accuracy = sum(target_test_ori==target_hat)/size(target_test_ori,1);
        fid_re_com = fopen('Output_2.txt','a+');
        fprintf(fid_re_com,strcat(...
            'Discriminant: ',...
            ', CodingName:',Mdl_Disc.CodingName,...
            ', recall:',num2str(Accuracy),...
            '\n'));
        fclose('all');
      
        
        
%         % optimized naivebayes model by matlab
%         
%         Mdl_Nai = fitcecoc(data_train_ori,target_train_ori,...
%             'Learners','NaiveBayes',...
%             'OptimizeHyperparameters','all',...
%             'HyperparameterOptimizationOptions',...
%             struct('AcquisitionFunctionName', 'expected-improvement-plus'));
%         target_hat = predict(Mdl_Nai, data_test_ori);
%         Accuracy = sum(target_test_ori==target_hat)/size(target_test_ori,1);
%         fid_re_com = fopen('Output_2.txt','a+');
%         fprintf(fid_re_com,strcat(...
%             'naivebayes: ',...
%             ', CodingName:',Mdl_Nai.CodingName,...
%             ', recall:',num2str(Accuracy),...
%             '\n'));
        
        % optimized naivebayes model by matlab
        
        Mdl_tree = fitcecoc(data_train_ori,target_train_ori,...
            'Learners','tree',...
            'OptimizeHyperparameters','all',...
            'HyperparameterOptimizationOptions',...
            struct('AcquisitionFunctionName', 'expected-improvement-plus'));
        target_hat = predict(Mdl_tree, data_test_ori);
        Accuracy = sum(target_test_ori==target_hat)/size(target_test_ori,1);
        fid_re_com = fopen('Output_2.txt','a+');
        fprintf(fid_re_com,strcat(...
            'naivebayes: ',...
            ', CodingName:',Mdl_tree.CodingName,...
            ', recall:',num2str(Accuracy),...
            '\n'));
        
        fclose('all');
        diary('Output_2.txt');
        Mdl_Disc.ModelParameters.BinaryLearners
%         Mdl_Nai.ModelParameters.BinaryLearners
        Mdl_tree.ModelParameters.BinaryLearners
        diary off;
              
end
        
                
fclose('all');             
movefile ('Output_2.txt' , strcat(filename_train, '\', filename_train,'_classificationCombined_2','.txt'));


end

function [distance_ori, target_ori]=dm2data(filename)

i = find(filename=='.');
if isempty(i) 
    l = length(filename);
    l_i = length(i);
    ex_train = filename(i(l_i)+1:l);
    if ~strcmp(ex_train, 'dm')
        error('It is not a correst dm filename.');
    end
else 
    error('It is not a correst dm filename.');
end

fidin = fopen(filename);
nline=0;
while ~feof(fidin) 
    tline=fgetl(fidin); 
    nline=nline+1;
if strcmp(tline, '.CLASS MEMBERSHIP')
    tline=fgetl(fidin); 
    target_ori = str2int(tline)';
    nline=nline+1;
end
if strcmp(tline, '.DISTANCE MATRIX')
    distance_ori = dlmread(filename, '	', nline,0);
    distance_ori = distance_ori(1:size(distance_ori,1),1:size(distance_ori,2)-1);
end
end
fclose(fidin);

end

function [data_ori, target_ori]=tsv2data(filename)

s = dlmread(filename);
data_ori = s(1:size(s,1),2:size(s,2));
target_ori  = s(1:size(s,1),1);

end

function [data_train_ori, target_train_ori] = simul_data_const(filename_train)

switch filename_train
    
    case 'fisheriris'
        load('fisheriris');
        data_train_ori = meas;
        labels_ori = unique(species);
        T = length(labels_ori);
        n = size(data_train_ori,1);
        target_train_ori = zeros(n, 1);
        for t = 1:T
            target_train_ori(strcmp(species, labels_ori(t))) = t;
        end
        
        
            
        
end


end