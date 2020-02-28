function [accuracy, model, Out_te, data_train_ori, target_train_ori, data_test_ori, target_test_ori, acc_std] = smds_main_accuracy_std(filename_train, dim, alpha, q, disttype, cla_method,varargin)
%  This code is to combine all possible situation.
%  [Accuracy, model, Out_te, data_train_ori, target_train_ori, data_test_ori, target_test_ori]
%  INPUTS:
%
%       filename_train: the name of training data file. This code is
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
%       cla_method: 
%% 

% Open a new file 'Output.txt' for recording.
fclose('all'); 
if exist('Output.txt') ==2; delete('Output.txt'); end
fid = fopen('Output.txt','a+');fclose(fid); 

% Identify the extensi  n(1)),'distance')
i = strfind(filename_train,'.');
if ~isempty(i) 
    ex_train = filename_train(i(end)+1:end);
else 
    ex_train = '';
end

switch ex_train

    case 'dm'

        [data_train_ori, target_train_ori] = dm2data(filename_train);

    case 'tsv'

        [data_train_ori, target_train_ori] = tsv2data(filename_train);

    case ''

        [data_train_ori, target_train_ori] = simul_data_const(filename_train);

    case 'dsv'

        [data_train_ori, target_train_ori, disttype ] = distance_dsv2data(filename_train);
end
delimiters_line = strfind(filename_train,'_');
dataname=filename_train(1:delimiters_line(1)-1);
% else 
%     data_train_ori = file_train(:,2:end);
%     target_train_ori = file_train(:,1);
%     mat_test_ori = cell2mat(varargin(2));
%     data_test_ori = mat_test_ori(:,2:end);
%     target_test_ori = mat_test_ori(:,1);
% end

% Construct training data matrix

     
% Construct testing data matrix   
if ismember('test',varargin)

    filename_test = char(varargin(find(ismember(varargin,'test'),1)+1));
    i = strfind(filename_test,'.');
    if ~isempty(i) 
        ex_test = filename_test(i(end)+1:end);
        if ~strcmp(ex_test, ex_train)
            error('This file does not match the training file.');
        else
            if ~strcmp(ex_test, 'tsv')&& ~strcmp(ex_test, 'dsv')
                error('This type of file cannot be treated as testing file.');
            end
        end
        
    else 
        error('Name of testing file should be with extension.');
    end
    if strcmp(ex_test, 'tsv')
        [data_test_ori, target_test_ori] = tsv2data(filename_test);
    end
    
    if strcmp(ex_test, 'dsv')
        [data_test_ori, target_test_ori, disttype_test] = distance_dsv2data(filename_test);
        if ~strcmp(disttype_test,disttype)
            error('Distance metrics of training and testing data are different.');
        end
    end

end
    
if ismember('test',varargin) && ismember('k-fold',varargin)
    error('This code can just do k-fold validation or testing for one time.');
end
   
k = '';
if ismember('k-fold',varargin) && ~ismember('distance',varargin)
%     k = cell2mat(varargin(find(ismember(varargin,'k-fold'),1)+1));
%    k=3;
%      k = min(min(hist(target_train_ori , unique(target_train_ori))),10);
%         k = 20;
     indices = dlmread(cell2mat(varargin(find(ismember(varargin,'k-fold'),1)+1)));
     k = max(indices);
        switch cla_method
            
            case '1vsall'

                if ~ismember('extrapar',varargin)
                   
                    [ Accuracy, model, Out_te ] = Kfoldtest_SCMD_1vsall( k, data_train_ori, ...
                        target_train_ori, dim, alpha, q, disttype, indices);
                else
%                     if nargin ==10 && strcmp(varargin(9),'extra')
                        par_extra = cell2struct(varargin(find(ismember(varargin,'extrapar'),1)+1));
                        [ Accuracy, model, Out_te ] = Kfoldtest_SCMD_1vsall( k, data_train_ori, ...
                            target_train_ori, dim, alpha, q, disttype, indices, 'extrapar',par_extra);
%                     else
%                         error('Some input arguments are not correct.')
%                     end
                
                end
                
            case '1vs1'
                
                if ~ismember('extrapar',varargin)
                    [ Accuracy, model, Out_te ] = Kfoldtest_SCMD_1vs1( k, data_train_ori, ...
                        target_train_ori, dim, alpha, q, disttype, indices);
                else
%                     if nargin ==10 && strcmp(varargin(9),'extra')
                        par_extra = cell2struct(varargin(find(ismember(varargin,'extrapar'),1)+1));
                        [ Accuracy, model, Out_te ] = Kfoldtest_SCMD_1vs1( k, data_train_ori, ...
                            target_train_ori, dim, alpha, q, disttype, indices, 'extrapar',par_extra);
%                     else
%                         error('Some input arguments are not correct.')
%                     end
                
                end
        end
        
        data_test_ori = data_train_ori;
        target_test_ori  =  target_train_ori;      
        accuracy = mean(Accuracy);          
        acc_std = std(Accuracy);
end     

if ismember('k-fold',varargin) && ismember('distance',varargin)
%     k = cell2mat(varargin(find(ismember(varargin,'k-fold'),1)+1));
%      k = min(min(hist(target_train_ori , unique(target_train_ori))),10);
%         k = 20;
     indices = dlmread(cell2mat(varargin(find(ismember(varargin,'k-fold'),1)+1)));
     k = max(indices);
        switch cla_method
            
            case '1vsall'

                if ~ismember('extrapar',varargin)
                   
                    [ Accuracy, model, Out_te ] = Kfoldtest_SCMD_1vsall( k, data_train_ori, ...
                        target_train_ori, dim, alpha, q, disttype, indices, 'distance');
                else
%                     if nargin ==10 && strcmp(varargin(9),'extra')
                        par_extra = cell2struct(varargin(find(ismember(varargin,'extrapar'),1)+1));
                        [ Accuracy, model, Out_te ] = Kfoldtest_SCMD_1vsall( k, data_train_ori, ...
                            target_train_ori, dim, alpha, q, disttype, indices, 'distance','extrapar',par_extra);
%                     else
%                         error('Some input arguments are not correct.')
%                     end
                
                end
                
            case '1vs1'
                
                if ~ismember('extrapar',varargin)
                    [ Accuracy, model, Out_te ] = Kfoldtest_SCMD_1vs1( k, data_train_ori, ...
                        target_train_ori, dim, alpha, q, disttype, indices,'distance');
                else
%                     if nargin ==10 && strcmp(varargin(9),'extra')
                        par_extra = cell2struct(varargin(find(ismember(varargin,'extrapar'),1)+1));
                        [ Accuracy, model, Out_te ] = Kfoldtest_SCMD_1vs1( k, data_train_ori, ...
                            target_train_ori, dim, alpha, q, disttype, indices, 'distance','extrapar',par_extra);
%                     else
%                         error('Some input arguments are not correct.')
%                     end
                
                end
        end
    data_test_ori = data_train_ori;
    target_test_ori  =  target_train_ori;                
    accuracy = mean(Accuracy);          
    acc_std = std(Accuracy);
 
end     

if ismember('test',varargin) && ismember('distance',varargin)
        
        switch cla_method
            
            case '1vsall'
                
                if ~ismember('extrapar',varargin)
                    [model,cutpoint] = SMDSTraining_1vsall(data_train_ori, ...
                        target_train_ori, data_test_ori ,target_test_ori, ...
                        dim, alpha, q, disttype, 'distance');
                     Out_te=SQREDM_SMDS_test_1vsall(model,cutpoint);
                else
%                     if nargin ==10 && strcmp(varargin(9),'extra')
                        par_extra = cell2struct(varargin(find(ismember(varargin,'extrapar'),1)+1));
                        [model,cutpoint] = SMDSTraining_1vsall(data_train_ori, ...
                             target_train_ori, data_test_ori ,target_test_ori, ...
                             dim, alpha,q, 'distance', 'extrapar', par_extra);
                         Out_te=SQREDM_SMDS_test_1vsall(model,cutpoint);
%                     else
%                         error('Some input arguments are not correct.')
%                     end
                
                end
                
                 
            case '1vs1'
                
                if ~ismember('extrapar',varargin)
                    [model,cutpoint] = SMDSTraining_1vs1(data_train_ori, ...
                        target_train_ori, data_test_ori ,target_test_ori, ...
                        dim, alpha, q, disttype, 'distance');
                     Out_te=SQREDM_SMDS_test_1vs1(model,cutpoint);
                else
%                     if nargin ==10 && strcmp(varargin(9),'extra')
                        par_extra = cell2struct(varargin(find(ismember(varargin,'extrapar'),1)+1));
                        [model,cutpoint] = SMDSTraining_1vs1(data_train_ori, ...
                             target_train_ori, data_test_ori ,target_test_ori, ...
                             dim, alpha,q, 'distance','extrapar', par_extra);
                         Out_te=SQREDM_SMDS_test_1vs1(model,cutpoint);
%                     else
%                         error('Some input arguments are not correct.')
%                     end
                
                end
       
        end       
                   
        Accuracy = Out_te.Accuracy;   
        accuracy = mean(Accuracy);          
        acc_std = std(Accuracy);
        
      
end


if ismember('test',varargin) && ~ismember('distance',varargin)
    
     switch cla_method
            
            case '1vsall'
                
                if ~ismember('extrapar',varargin)
                    [model,cutpoint] = SMDSTraining_1vsall(data_train_ori, ...
                        target_train_ori, data_test_ori ,target_test_ori, ...
                        dim, alpha, q, disttype);
                     Out_te=SQREDM_SMDS_test_1vsall(model,cutpoint);
                else
%                     if nargin ==10 && strcmp(varargin(9),'extra')
                        par_extra = cell2struct(varargin(find(ismember(varargin,'extrapar'),1)+1));
                        [model,cutpoint] = SMDSTraining_1vsall(data_train_ori, ...
                             target_train_ori, data_test_ori ,target_test_ori, ...
                             dim, alpha,q, 'extrapar', par_extra);
                         Out_te=SQREDM_SMDS_test_1vsall(model,cutpoint);
%                     else
%                         error('Some input arguments are not correct.')
%                     end
                
                end
                
                 
            case '1vs1'
                
                if ~ismember('extrapar',varargin)
                    [model,cutpoint] = SMDSTraining_1vs1(data_train_ori, ...
                        target_train_ori, data_test_ori ,target_test_ori, ...
                        dim, alpha, q, disttype);
                     Out_te=SQREDM_SMDS_test_1vs1(model,cutpoint);
                else
%                     if nargin ==10 && strcmp(varargin(9),'extra')
                        par_extra = cell2struct(varargin(find(ismember(varargin,'extrapar'),1)+1));
                        [model,cutpoint] = SMDSTraining_1vs1(data_train_ori, ...
                             target_train_ori, data_test_ori ,target_test_ori, ...
                             dim, alpha,q,'extrapar', par_extra);
                         Out_te=SQREDM_SMDS_test_1vs1(model,cutpoint);
%                     else
%                         error('Some input arguments are not correct.')
%                     end
                
                end
       
     end       
                   
        Accuracy = Out_te.Accuracy;   
        accuracy = mean(Accuracy);          
        acc_std = std(Accuracy);
end

        
                
                
            

             
        
     
fid = fopen('Output.txt','a+');
fprintf(fid,'\n\n--------------------------------------------------------------\n');
fprintf(fid,'                  Overall Accuracy Rate: %3d                  \n',mean(Accuracy));
fprintf(fid,'--------------------------------------------------------------\n\n\n');
fprintf('Overall Accuracy Rate: %3d  \n',mean(Accuracy));
fclose('all');
figure;scatter(model(1).Out.X(1,:),model(1).Out.X(2,:),[],model(1).pars_2c.PY);
title(strcat(dataname,', SMDS:','dim:',num2str(dim),...
    ', alpha:',num2str(alpha),', q:',num2str(q),', dist:',disttype,...
    ', cla_method:',cla_method,', lable_a:', num2str(model(1).lable_a),...
    ', lable_b:', num2str(model(1).lable_b) ,', recall:',num2str(mean(Accuracy),...
    '%1.4f')));
if exist(dataname,'dir')~=7
    mkdir(dataname) ;
end

% try
%     saveas(gcf, strcat('Results_extended\',dataname, '\', dataname,'_SMDS','_dim_',num2str(dim),...
%         '_alpha_',num2str(alpha),'_q_',num2str(q),'_dist_',disttype(1:3),...
%         '_',cla_method,'_',char(varargin(find(ismember(varargin,'test'),1))),...
%         '_',char(varargin(find(ismember(varargin,'k-fold'),1))),...
%         num2str(k),'_recall_',num2str(mean(Accuracy),...
%         '%1.4f'),'.png'));
%     close;
% end

try
    saveas(gcf, strcat('Results\',dataname, '\', dataname,'_SMDS','_dim_',num2str(dim),...
        '_alpha_',num2str(alpha),'_q_',num2str(q),'_dist_',disttype(1:3),...
        '_',cla_method,'_',char(varargin(find(ismember(varargin,'test'),1))),...
        '_',char(varargin(find(ismember(varargin,'k-fold'),1))),...
        num2str(k),'_recall_',num2str(mean(Accuracy),...
        '%1.4f'),'.png'));
    close;
end

% movefile ('Output.txt' , strcat('Results_extended\', dataname, '\', dataname,'_SMDS','_d',num2str(dim),...
%     '_al',num2str(alpha),'_q',num2str(q),'_',disttype(1:3),...
%     '_',cla_method,'_',char(varargin(find(ismember(varargin,'test'),1))),...
%     '_',char(varargin(find(ismember(varargin,'k-fold'),1))),...
%     num2str(k),'_re_',num2str(mean(Accuracy),...
%     '%1.4f'),'.txt'));

movefile ('Output.txt' , strcat('Results\', dataname, '\', dataname,'_SMDS','_d',num2str(dim),...
    '_al',num2str(alpha),'_q',num2str(q),'_',disttype(1:3),...
    '_',cla_method,'_',char(varargin(find(ismember(varargin,'test'),1))),...
    '_',char(varargin(find(ismember(varargin,'k-fold'),1))),...
    num2str(k),'_re_',num2str(mean(Accuracy),...
    '%1.4f'),'.txt'));


% fid_re = fopen(strcat('Results_extended\', dataname, '\',dataname,'_results', '.txt'),'a+');
fid_re = fopen(strcat('Results\', dataname, '\',dataname,'_results', '.txt'),'a+');


fprintf(fid_re,strcat('dim:',num2str(dim),...
    ', alpha:',num2str(alpha),', q:',num2str(q),', dist:',disttype,...
    ', cla_method:',cla_method,',   ',char(varargin(find(ismember(varargin,'test'),1))),...
    ': ',char(varargin(find(ismember(varargin,'test'),1)+1)),...
    ',  ',char(varargin(find(ismember(varargin,'k-fold'),1))),...
    ': ',num2str(k),', recall:',num2str(mean(Accuracy),...
    '%1.4f'),'\n'));
fclose('all');

end

% function errorate = SMDStesting(data_train_ori,target_train_ori, data_test_ori ,target_test_ori, dim, alpha, q, disttype)
%                     
% [model,cutpoint] = SMDSTraining_1vs1(data_train_ori, ...
%                         target_train_ori, data_test_ori ,target_test_ori, dim, alpha, q, disttype);
% Out_te=SQREDM_SMDS_test_1vs1(model,cutpoint);
% errorate=1-Out_te.Accuracy;
% end




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

function [data_train_ori, target_train_ori] = simul_data_const(filename)

switch filename
    
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

function [data_ori, target_ori,  disttype_name ] = distance_dsv2data(filename)

disttype={'euclidean','squaredeuclidean',...
    'seuclidean','cityblock','minkowski','chebychev','correlation','cosine',...
    'hamming','jaccard','spearman','dtw'};
filename_split = strsplit(filename,{'_','.'});
if sum(ismember(filename_split,disttype))~=1
    error('The format of distance filename can not be recognized')
else
    disttype_name = char(filename_split(ismember(filename_split,disttype)));
    if strcmp(disttype_name, 'dtw')
        disttype_name = 'dtw_dist';
    end
end

mat_ori = dlmread(filename);
data_ori = mat_ori(:,2:end);
target_ori = mat_ori(:,1);

end
