function proceed =SMDS_results_gridsearch_forBound_Simulateddata(datasets,path)
rng default
% alpha = optimizableVariable('alpha',[1,9],'Type','integer');
% q = optimizableVariable('q',[-25,10],'Type','integer');
% disttype = optimizableVariable('disttype',{'euclidean','squaredeuclidean',...
%     'seuclidean','cityblock','minkowski','chebychev','correlation',...
%     'hamming','spearman','dtw_dist'},'Type','categorical');
% cla_method = optimizableVariable('cla_method',{'1vs1','1vsall'},'Type','categorical');
% dim = optimizableVariable('dim',[3,7],'Type','integer');

n_da = length(datasets);


% grid search for different distance types and classification straegies(one-vs.-all and one-vs.-one)
% disttypes = {'euclidean','squaredeuclidean',...
%     'seuclidean','cityblock','minkowski','chebychev','correlation',...
%     'hamming','spearman','dtw_dist'};
% disttypes = {'euclidean','squaredeuclidean',...
%     'seuclidean','cityblock','minkowski','chebychev','correlation',...
%     'spearman','dtw_dist'};
% disttypes = {'dtw_dist'};
disttypes = {'euclidean','squaredeuclidean',...
    'seuclidean','cityblock','minkowski','chebychev',...
    'dtw_dist'};
n_di = length(disttypes);
d_cl_results(1:n_di,1) = disttypes;
d_cl_results(1:n_di,2) = {'1vsall'};
d_cl_results(n_di+1:2*n_di,1) = disttypes;
d_cl_results(n_di+1:2*n_di,2) = {'1vs1'};
addpath(path);
for d= 1: n_da
    dataset = char(datasets(d));
    result_folder = strcat('Results\',dataset);
%     result_folder = strcat('Results_extended\',dataset);
    if exist(result_folder,'dir')~=7
        mkdir(result_folder) ;
    end
    filename_train = strcat(dataset,'_TRAIN.ssv');
    filename_test = strcat(dataset,'_TEST.ssv');
    filename_path = strcat(path,'\',filename_train);
     if exist(filename_path,'file')~=2
        error('The datafile is not correct.')
    end
    addpath(result_folder);
    
   
    % filename_test = strcat(dataset,'_TEST.tsv');
    
    [data_train_ori, target_train_ori]=ssv2data(filename_train);
    [data_test_ori, target_test_ori]=ssv2data(filename_test);
 
%   data extention   
%     data_extended = [data_train_ori, target_train_ori];
%     n_n = size(data_extended,1);
%     data_extended = datasample(data_extended,5*n_n);
%     data_train_ori = data_extended(:,1:end-1);
%     target_train_ori = data_extended(:,end);
    
    k = min(min(hist(target_train_ori , unique(target_train_ori))),10);
    indices = crossvalind('Kfold',target_train_ori,k);
    indicesfile_name_10 = strcat(dataset,'_k-fold_indices_k_',num2str(k),'.csv');
    csvwrite(strcat(result_folder, '\', dataset,'_k-fold_indices_k_',num2str(k),'.csv'),indices);
    k = min(min(hist(target_train_ori , unique(target_train_ori))),5);
    indices = crossvalind('Kfold',target_train_ori,k);
    indicesfile_name_5 = strcat(dataset,'_k-fold_indices_k_',num2str(k),'.csv');
    csvwrite(strcat(result_folder, '\', dataset,'_k-fold_indices_k_',num2str(k),'.csv'),indices);
    
    for i = 1:length(d_cl_results)
        disttype = char(d_cl_results(i,1));
        cla_method = char(d_cl_results(i,2));
         if exist(strcat(result_folder, '\', dataset,'_',disttype,...
            '_train.dsv'),'file') ~= 2
            [data_train_distance, ~, data_test_distance] = ...
                EDM_PARConstruction(data_train_ori, ...
                target_train_ori, data_test_ori, ...
                target_test_ori, disttype);
            csvwrite(strcat(result_folder, '\', dataset,'_',disttype,...
                '_train.dsv'),[target_train_ori,data_train_distance]);

            csvwrite(strcat(result_folder, '\', dataset,'_',disttype,...
                '_test.dsv'),[target_test_ori,data_test_distance]);
        
        end

        filename_distance_train = strcat(dataset,'_',disttype,...
            '_train.dsv');
    
        
        [accuracy_1, ~] = smds_main_accuracy_kfold_distance_accuracy_std(...
            filename_distance_train, 5, 0.8, -2, ...
            char(d_cl_results(i,1)),...
            char(d_cl_results(i,2)),indicesfile_name_10);
        [accuracy_2, ~] = smds_main_accuracy_kfold_distance_accuracy_std(...
            filename_distance_train, 5, 0.8, -2, ...
            char(d_cl_results(i,1)),...
            char(d_cl_results(i,2)),indicesfile_name_5);
        d_cl_results(i,3) = {accuracy_1+accuracy_2};
    end
    
    d_cl_opt = ...
        d_cl_results(...
            find(cell2mat(d_cl_results(:,3))==max(cell2mat(d_cl_results(:,3)))),:);
%         find(cell2mat(d_cl_results(:,3))<=quantile(cell2mat(d_cl_results(:,3)),0.1)),:);
    
    
   
 
    for l =1:length(d_cl_opt(:,1))
        
        disttype = char(d_cl_opt(l,1));
        cla_method = char(d_cl_opt(l,2));
        if exist(strcat(result_folder, '\', dataset,'_',disttype,...
            '_train.dsv'),'file') ~= 2
            [data_train_distance, ~, data_test_distance] = ...
                EDM_PARConstruction(data_train_ori, ...
                target_train_ori, data_test_ori, ...
                target_test_ori, disttype);
            csvwrite(strcat(result_folder, '\',dataset,'_',disttype,...
                '_train.dsv'),[target_train_ori,data_train_distance]);

            csvwrite(strcat(result_folder, '\', dataset,'_',disttype,...
                '_test.dsv'),[target_test_ori,data_test_distance]);
        
        end
        
        filename_distance_train = strcat(dataset,'_',disttype,...
            '_train.dsv');
        filename_distance_test = strcat(dataset,'_',disttype,...
            '_test.dsv');

        i=1;j=1
        for al = 0.1:0.1:0.9
        
            j=1;k=0;
            for q = -2.5 : 0.1: 0
            

                [accuracy, ~, Out_te, ~, ~, ~, target_test, ~]  = ...
                    smds_main_accuracy_std(filename_distance_train, 5, al, q, ...
                    disttype, cla_method, 'distance', 'test',...
                    filename_distance_train);
                eval(strcat(dataset,'_',disttype,'_',cla_method,...
                    '_selftest_accuracy(i,j) = accuracy;'));
                eval(strcat(dataset,'_',disttype,'_',cla_method,...
                    '_selftest_F_crits(1:size(Out_te.PY_te_F_crit,1),k+1:k+size(Out_te.PY_te_F_crit,2),i) = Out_te.PY_te_F_crit;'));
                j = j+1;
                k = k + size(Out_te.PY_te_F_crit,2);
            end
            i = i+1
        end
        eval(strcat(dataset,'_',disttype,'_selftest_target = target_test;'));
        
        i=1;j=1
        for al = 0.1:0.1:0.9
            i
            j=1;k=0;
            for q = -2.5 : 0.1: 0
            
                [accuracy, ~, Out_te, ~, ~, ~, target_test, ~]  = ...
                    smds_main_accuracy_std(filename_distance_train, 5, al, q, ...
                    disttype, cla_method, 'distance', 'test',...
                    filename_distance_test);
                eval(strcat(dataset,'_',disttype,'_',cla_method,...
                    '_test_accuracy(i,j) = accuracy;'));
                eval(strcat(dataset,'_',disttype,'_',cla_method,...
                    '_test_F_crits(1:size(Out_te.PY_te_F_crit,1),k+1:k+size(Out_te.PY_te_F_crit,2),i) = Out_te.PY_te_F_crit;'));
                j = j+1;
                k = k + size(Out_te.PY_te_F_crit,2);
            end
            i = i+1;
        end
        eval(strcat(dataset,'_',disttype,'_test_target = target_test;'));
        
        i=1;j=1
        for al = 0.1:0.1:0.9
        
            j=1;
            for q = -2.5 : 0.1: 0
            
                [accuracy, ~, ~, ~, ~, ~, ~, std]  = ...
                    smds_main_accuracy_std(filename_distance_train, 5, al, q, ...
                    disttype, cla_method, 'distance', 'k-fold',...
                    indicesfile_name_5);
                eval(strcat(dataset,'_',disttype,'_',cla_method,...
                    '_k_fold_5_accuracy(i,j) = accuracy;'));
                eval(strcat(dataset,'_',disttype,'_',cla_method,...
                    '_k_fold_5_std(i,j) = std;'));
                j = j+1;
                
            end
            i = i+1
        end
        
        i=1;j=1
        for al = 0.1:0.1:0.9
        
            j=1;
            for q = -2.5 : 0.1: 0
            
                [accuracy, ~, ~, ~, ~, ~, ~, std]  = ...
                    smds_main_accuracy_std(filename_distance_train, 5, al, q, ...
                    disttype, cla_method, 'distance', 'k-fold',...
                    indicesfile_name_10);
                eval(strcat(dataset,'_',disttype,'_',cla_method,...
                    '_k_fold_10_accuracy(i,j) = accuracy;'));
                eval(strcat(dataset,'_',disttype,'_',cla_method,...
                    '_k_fold_10_std(i,j) = std;'));
                j = j+1;
                
            end
            i = i+1
        end
        
        save(strcat(result_folder,'\',dataset,'_',disttype,'_',cla_method,'.mat'), ...
            strcat(dataset,'_',disttype,'_',cla_method,...
                    '_selftest_accuracy'),...
            strcat(dataset,'_',disttype,'_',cla_method,'_selftest_F_crits'),...
            strcat(dataset,'_',disttype,'_',cla_method,...
                    '_test_accuracy'),...
            strcat(dataset,'_',disttype,'_',cla_method,'_test_F_crits'),...
            strcat(dataset,'_',disttype,'_',cla_method,...
                    '_k_fold_5_accuracy'),...
            strcat(dataset,'_',disttype,'_',cla_method,...
                    '_k_fold_5_std'),...
            strcat(dataset,'_',disttype,'_',cla_method,...
                    '_k_fold_10_accuracy'),...
            strcat(dataset,'_',disttype,'_',cla_method,...
                    '_k_fold_10_std'));       
        

    end
  
    
    proceed = d;

end

end

function [errorrate, std] = smds_main_errorrate_kfold_accuracy_std(filename_train, dim, alpha, q, disttype, cla_method, indicesfile_name)
[errorrate,~, ~, ~, ~, ~, ~, std] = smds_main_accuracy_std(filename_train, dim, alpha, q, disttype,cla_method,'k-fold',indicesfile_name);
end

function [errorrate, std] = smds_main_accuracy_kfold_distance_accuracy_std(filename_train, dim, alpha, q, disttype,cla_method,indicesfile_name )
[errorrate,~, ~, ~, ~, ~, ~, std] = smds_main_accuracy_std(filename_train, dim, alpha, q, disttype,cla_method,'distance','k-fold', indicesfile_name);
end

function errorrate = smds_main_accuracy_test_accuracy_std(filename_train, dim, alpha, q, disttype, cla_method, filename_test)
[errorrate,~, ~, ~, ~, ~, ~] = smds_main_accuracy_std(filename_train, dim, alpha, q, disttype,cla_method,'test',filename_test);
end

function [data_ori, target_ori]=ssv2data(filename)

s = dlmread(filename);
data_ori = s(1:size(s,1),2:size(s,2));
target_ori  = s(1:size(s,1),1);

end

function [data_ori, target_ori] = LoadUCRArchiveData()



end
