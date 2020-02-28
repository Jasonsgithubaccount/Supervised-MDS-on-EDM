function proceed =SMDS_results_gridsearch_forExtendedURCdata(datasets,path,kn)
% This is to performe the gridsearch of hyperparameter tunning
% alpha = optimizableVariable('alpha',[1,9],'Type','integer');
% q = optimizableVariable('q',[-25,10],'Type','integer');
% disttype = optimizableVariable('disttype',{'euclidean','squaredeuclidean',...
%     'seuclidean','cityblock','minkowski','chebychev','correlation',...
%     'hamming','spearman','dtw_dist'},'Type','categorical');
% cla_method = optimizableVariable('cla_method',{'1vs1','1vsall'},'Type','categorical');
% dim = optimizableVariable('dim',[3,7],'Type','integer');
rng default
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


for d= 1: n_da
    dataset = char(datasets(d));
    result_folder = strcat('Results_extendeddata\',dataset);
%     result_folder = strcat('Results_extended\',dataset);
    if exist(result_folder,'dir')~=7
        mkdir(result_folder) ;
    end
    data_folder = strcat(path,'\',dataset);
     if exist(data_folder,'dir')~=7
        error('The path is not correct.')
    end
    addpath(result_folder);
    addpath(data_folder);
    filename_train = strcat(dataset,'_TRAIN.tsv');
    filename_test = strcat(dataset,'_TEST.tsv');
    
    [data_train_ori, target_train_ori]=tsv2data(filename_train);
    [data_test_ori, target_test_ori]=tsv2data(filename_test);
 
%   data extention   
%     data_extended = [data_train_ori, target_train_ori];
%     n_n = size(data_extended,1);
%     data_extended = datasample(data_extended,5*n_n);
%     data_train_ori = data_extended(:,1:end-1);
%     target_train_ori = data_extended(:,end);
    
    for i = 1:length(d_cl_results)
        disttype = char(d_cl_results(i,1));
         [target_train_ori,data_train_ori] = ...
                UcrDataExtension_knn({dataset}, path, 'K', kn, 'Distance', disttype);
            
          k = min(min(hist(target_train_ori , unique(target_train_ori))),3);
            if exist(strcat(dataset,'_extended_k-fold_indices_k_',num2str(k),'.csv'),'file') ~= 2
                indices = crossvalind('Kfold',target_train_ori,k);
                indicesfile_name_3 = strcat(dataset,'_extended_k-fold_indices_k_',num2str(k),'.csv');
                csvwrite(strcat(result_folder, '\', dataset,'_extended_k-fold_indices_k_',num2str(k),'.csv'),indices);
            else
                indicesfile_name_3 = strcat(dataset,'_extended_k-fold_indices_k_',num2str(k),'.csv');
            end
            k = min(min(hist(target_train_ori , unique(target_train_ori))),5);
            if exist(strcat(dataset,'_extended_k-fold_indices_k_',num2str(k),'.csv'),'file') ~= 2    
                indices = crossvalind('Kfold',target_train_ori,k);
                indicesfile_name_5 = strcat(dataset,'_extended_k-fold_indices_k_',num2str(k),'.csv');
                csvwrite(strcat(result_folder, '\', dataset,'_extended_k-fold_indices_k_',num2str(k),'.csv'),indices);
            else
                indicesfile_name_5 = strcat(dataset,'_extended_k-fold_indices_k_',num2str(k),'.csv');
            end
        if exist(strcat(result_folder, '\', dataset,'_',disttype,...
            '_train_extended_knn_',num2str(kn),'.dsv'),'file') ~= 2
           
            
           
            
            [data_train_distance, ~, data_test_distance] = ...
                EDM_PARConstruction(data_train_ori, ...
                target_train_ori, data_test_ori, ...
                target_test_ori, disttype);
            csvwrite(strcat(result_folder, '\', dataset,'_',disttype,...
                '_train_extended_knn_',num2str(kn),'.dsv'),[target_train_ori,data_train_distance]);

            csvwrite(strcat(result_folder, '\', dataset,'_',disttype,...
                '_test_extended_knn_',num2str(kn),'.dsv'),[target_test_ori,data_test_distance]);
       
        end

        filename_distance_train = strcat(dataset,'_',disttype,...
            '_train_extended_knn_',num2str(kn),'.dsv');
        
        errorrate_1 = smds_main_errorrate_kfold_distance(...
            filename_distance_train, 3, 0.8, -2, ...
            char(d_cl_results(i,1)),...
            char(d_cl_results(i,2)),indicesfile_name_3);
%         errorrate_2 = smds_main_errorrate_kfold_distance(...
%             filename_distance_train, 5, 0.8, -2, ...
%             char(d_cl_results(i,1)),...
%             char(d_cl_results(i,2)),indicesfile_name_5);
        d_cl_results(i,3) = {errorrate_1};
    end
    
    d_cl_opt = ...
        d_cl_results(...
            find(cell2mat(d_cl_results(:,3))==min(cell2mat(d_cl_results(:,3)))),:);
%         find(cell2mat(d_cl_results(:,3))<=quantile(cell2mat(d_cl_results(:,3)),0.1)),:);
    
    
   
 
    for l =1:length(d_cl_opt(:,1))
        
        disttype = char(d_cl_opt(l,1));
        cla_method = char(d_cl_opt(l,2));
        if exist(strcat(result_folder, '\', dataset,'_',disttype,...
            '_train_extended_knn_',num2str(kn),'.dsv'),'file') ~= 2
            [data_train_distance, ~, data_test_distance] = ...
                EDM_PARConstruction(data_train_ori, ...
                target_train_ori, data_test_ori, ...
                target_test_ori, disttype);
            csvwrite(strcat(result_folder, '\',dataset,'_',disttype,...
                '_train_extended_knn_',num2str(kn),'.dsv'),[target_train_ori,data_train_distance]);

            csvwrite(strcat(result_folder, '\', dataset,'_',disttype,...
                '_test_extended_knn_',num2str(kn),'.dsv'),[target_test_ori,data_test_distance]);
        
        end
        
        filename_distance_train = strcat(dataset,'_',disttype,...
            '_train_extended_knn_',num2str(kn),'.dsv');
        filename_distance_test = strcat(dataset,'_',disttype,...
            '_test_extended_knn_',num2str(kn),'.dsv');
        
        
        i=1;j=1
        for al = 0.1:0.1:0.9
        
            j=1;k=0;
            for q = -2.5 : 0.1: 0
            

                [errorrate, ~, Out_te, ~, ~, ~, target_test]  = ...
                    smds_main(filename_distance_train, 3, al, q, ...
                    disttype, cla_method, 'distance', 'test',...
                    filename_distance_train);
                eval(strcat(dataset,'_',disttype,'_',cla_method,...
                    '_ext_knn_',num2str(kn),'_selftest(i,j) = errorrate;'));
                eval(strcat(dataset,'_',disttype,'_',cla_method,...
                    '_ext_knn_',num2str(kn),'_selftest_F_crits(1:size(Out_te.PY_te_F_crit,1),k+1:k+size(Out_te.PY_te_F_crit,2),i) = Out_te.PY_te_F_crit;'));
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
            
                [errorrate, ~, Out_te, ~, ~, ~, target_test]  = ...
                    smds_main(filename_distance_train, 3, al, q, ...
                    disttype, cla_method, 'distance', 'test',...
                    filename_distance_test);
                eval(strcat(dataset,'_',disttype,'_',cla_method,...
                    '_ext_knn_',num2str(kn),'_test(i,j) = errorrate;'));
                eval(strcat(dataset,'_',disttype,'_',cla_method,...
                    '_ext_knn_',num2str(kn),'_test_F_crits(1:size(Out_te.PY_te_F_crit,1),k+1:k+size(Out_te.PY_te_F_crit,2),i) = Out_te.PY_te_F_crit;'));
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
            
                [errorrate, ~, ~, ~, ~, ~, ~]  = ...
                    smds_main(filename_distance_train, 3, al, q, ...
                    disttype, cla_method, 'distance', 'k-fold',...
                    indicesfile_name_5);
                eval(strcat(dataset,'_',disttype,'_',cla_method,...
                    '_ext_knn_',num2str(kn),'_k_fold5(i,j) = errorrate;'));
                j = j+1;
                
            end
            i = i+1
        end
        
        i=1;j=1
        for al = 0.1:0.1:0.9
        
            j=1;
            for q = -2.5 : 0.1: 0
            
                [errorrate, ~, ~, ~, ~, ~, ~]  = ...
                    smds_main(filename_distance_train, 3, al, q, ...
                    disttype, cla_method, 'distance', 'k-fold',...
                    indicesfile_name_3);
                eval(strcat(dataset,'_',disttype,'_',cla_method,...
                    '_ext_knn_',num2str(kn),'_d_3_k_fold3(i,j) = errorrate;'));
                j = j+1;
                
            end
            i = i+1
        end
        
        save(strcat(result_folder,'\',dataset,'_',disttype,'_',cla_method,'.mat'), ...
            strcat(dataset,'_',disttype,'_',cla_method,'_ext_knn_',num2str(kn),'_selftest'),...
            strcat(dataset,'_',disttype,'_',cla_method,'_ext_knn_',num2str(kn),'_selftest_F_crits'),...
            strcat(dataset,'_',disttype,'_',cla_method,'_ext_knn_',num2str(kn),'_test'),...
            strcat(dataset,'_',disttype,'_',cla_method,'_ext_knn_',num2str(kn),'_test_F_crits'),...
            strcat(dataset,'_',disttype,'_',cla_method,'_ext_knn_',num2str(kn),'_d_3_k_fold5'),...
            strcat(dataset,'_',disttype,'_',cla_method,'_ext_knn_',num2str(kn),'_d_3_k_fold3'));       
        
%         save(strcat(result_folder,'\',dataset,'_',disttype,'_',cla_method,'_extended','.mat'), ...
%             strcat(dataset,'_',disttype,'_',cla_method,'_selftest'),...
%             strcat(dataset,'_',disttype,'_',cla_method,'_selftest_F_crits'),...
%             strcat(dataset,'_',disttype,'_',cla_method,'_test'),...
%             strcat(dataset,'_',disttype,'_',cla_method,'_test_F_crits'),...
%             strcat(dataset,'_',disttype,'_',cla_method,'_k_fold'));
       
    end
  
    
    proceed = d;

end

end

function errorrate = smds_main_errorrate_kfold(filename_train, dim, alpha, q, disttype, cla_method, indicesfile_name)
[errorrate,~, ~, ~, ~, ~, ~] = smds_main(filename_train, dim, alpha, q, disttype,cla_method,'k-fold',indicesfile_name);
end

function errorrate = smds_main_errorrate_kfold_distance(filename_train, dim, alpha, q, disttype,cla_method,indicesfile_name )
[errorrate,~, ~, ~, ~, ~, ~] = smds_main(filename_train, dim, alpha, q, disttype,cla_method,'distance','k-fold', indicesfile_name);
end

function errorrate = smds_main_errorrate_test(filename_train, dim, alpha, q, disttype, cla_method, filename_test)
[errorrate,~, ~, ~, ~, ~, ~] = smds_main(filename_train, dim, alpha, q, disttype,cla_method,'test',filename_test);
end

function [data_ori, target_ori]=tsv2data(filename)

s = dlmread(filename);
data_ori = s(1:size(s,1),2:size(s,2));
target_ori  = s(1:size(s,1),1);

end

function [data_ori, target_ori] = LoadUCRArchiveData()



end
