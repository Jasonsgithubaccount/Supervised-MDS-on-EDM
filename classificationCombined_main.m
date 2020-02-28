function [Accuracy, data_train_ori, target_train_ori, data_test_ori, target_test_ori] = classificationCombined_main(filename_train, k, varargin)
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
        
    case 'ssv'
        
        [data_train_ori, target_train_ori] = tsv2data(filename_train);
        
    case ''
        
        [data_train_ori, target_train_ori] = simul_data_const(filename_train);
        
end

delimiters_line = strfind(filename_train,'_');
delimiters_slash = strfind(filename_train,'\');
dataname=filename_train(delimiters_slash(end)+1:delimiters_line(end)-1);
        
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
            switch ex_test
                
                case 'tsv'
                    [data_test_ori, target_test_ori] = tsv2data(filename_test);
                    
                case 'ssv'
                    [data_test_ori, target_test_ori] = ssv2data(filename_test);
                    
                otherwise
                    error('This type of file cannot be treated as testing file.');
            
            end
        end
        
    else 
        error('Name of testing file should be with extension.');
    end
   

end
    

        
% optimized KNN model by matlab

Mdl_KNN = fitcecoc(data_train_ori,target_train_ori,...
    'Learners','knn',...
    'OptimizeHyperparameters','all',...
    'HyperparameterOptimizationOptions',...
    struct('AcquisitionFunctionName', 'expected-improvement-plus', 'KFold',k));
target_hat = predict(Mdl_KNN, data_test_ori);
Accuracy = sum(target_test_ori==target_hat)/size(target_test_ori,1);
fid_re_com = fopen('Output_2.txt','a+');
fprintf(fid_re_com,strcat(...
    'KNN: ',...
    'Distance:',Mdl_KNN.BinaryLearners{1, 1}.ModelParameters.Distance,...
    ', DistanceWeight:',Mdl_KNN.BinaryLearners{1, 1}.ModelParameters.DistanceWeight,...
    ', NumNeighbors:',num2str(Mdl_KNN.BinaryLearners{1, 1}.ModelParameters.NumNeighbors),...
    ', CodingName:',Mdl_KNN.CodingName,...
    ', StandardizeData:',num2str(Mdl_KNN.BinaryLearners{1, 1}.ModelParameters.StandardizeData),...
    ', MinObjective:',num2str(Mdl_KNN.HyperparameterOptimizationResults.MinObjective),...
    ', KFold:',num2str(k),...
    ', recall:',num2str(Accuracy),...
    '\n'));
fclose('all');



% optimized SVM model by matlab

Mdl_SVM = fitcecoc(data_train_ori,target_train_ori,...
    'Learners','svm',...
    'OptimizeHyperparameters','all',...
    'HyperparameterOptimizationOptions',...
    struct('AcquisitionFunctionName', 'expected-improvement-plus', 'KFold',k));
target_hat = predict(Mdl_SVM, data_test_ori);
Accuracy = sum(target_test_ori==target_hat)/size(target_test_ori,1);
fid_re_com = fopen('Output_2.txt','a+');
fprintf(fid_re_com,strcat(...
    'SVM: ',...
    'Kernel Function:',Mdl_SVM.BinaryLearners{1, 1}.KernelParameters.Function,...
    ', CodingName:',Mdl_SVM.CodingName,...
    ', MinObjective:',num2str(Mdl_KNN.HyperparameterOptimizationResults.MinObjective),...
    ', KFold:',num2str(k),...
    ', recall:',num2str(Accuracy),...
    '\n'));
fclose('all');


% optimized discriminant model by matlab
        
Mdl_Disc = fitcecoc(data_train_ori,target_train_ori,...
    'Learners','discriminant',...
    'OptimizeHyperparameters','all',...
    'HyperparameterOptimizationOptions',...
    struct('AcquisitionFunctionName', 'expected-improvement-plus', 'KFold',k));
target_hat = predict(Mdl_Disc, data_test_ori);
Accuracy = sum(target_test_ori==target_hat)/size(target_test_ori,1);
fid_re_com = fopen('Output_2.txt','a+');
fprintf(fid_re_com,strcat(...
    'Discriminant: ',...
    ', CodingName:',Mdl_Disc.CodingName,...
    ', MinObjective:',num2str(Mdl_KNN.HyperparameterOptimizationResults.MinObjective),...
    ', KFold:',num2str(k),...
    ', recall:',num2str(Accuracy),...
    '\n'));
fclose('all');
      
        

        
% optimized tree model by matlab

Mdl_tree = fitcecoc(data_train_ori,target_train_ori,...
    'Learners','tree',...
    'OptimizeHyperparameters','all',...
    'HyperparameterOptimizationOptions',...
    struct('AcquisitionFunctionName', 'expected-improvement-plus', 'KFold',k));
target_hat = predict(Mdl_tree, data_test_ori);
Accuracy = sum(target_test_ori==target_hat)/size(target_test_ori,1);
fid_re_com = fopen('Output_2.txt','a+');
fprintf(fid_re_com,strcat(...
    'naivebayes: ',...
    ', CodingName:',Mdl_tree.CodingName,...
    ', MinObjective:',num2str(Mdl_KNN.HyperparameterOptimizationResults.MinObjective),...
    ', KFold:',num2str(k),...
    ', recall:',num2str(Accuracy),...
    '\n'));

fclose('all');
diary('Output_2.txt');
Mdl_KNN.ModelParameters.BinaryLearners
Mdl_SVM.ModelParameters.BinaryLearners
Mdl_Disc.ModelParameters.BinaryLearners
Mdl_tree.ModelParameters.BinaryLearners
diary off;

                
fclose('all'); 
if exist(strcat('Results_classificationCombined\',dataname),'dir')~=7
    mkdir(strcat('Results_classificationCombined\',dataname)) ;
end
movefile ('Output_2.txt' , strcat('Results_classificationCombined\',dataname, '\', dataname,'_classificationCombined','_KFold_',num2str(k),'.txt'));


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

function [data_ori, target_ori]=ssv2data(filename)

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