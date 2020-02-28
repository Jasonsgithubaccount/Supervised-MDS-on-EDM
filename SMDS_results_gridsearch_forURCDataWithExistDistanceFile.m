function proceed =SMDS_results_gridsearch_forURCDataWithExistDistanceFile(datasets,disttypes, path)
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
% disttypes = {'DTWR1','DTWR02','DTWR05','DTWR08','DDTWR1','DDTWR02','DDTWR05','DDTWR08','LCSS','ERP','MSM','TWE'};
% disttypes = {'euclidean','squaredeuclidean',...
%     'seuclidean','cityblock','minkowski','chebychev','correlation',...
%     'spearman','dtw_dist'};
% disttypes = {'dtw_dist'};
% disttypes = {'euclidean','squaredeuclidean',...
%     'seuclidean','cityblock','minkowski','chebychev',...
%     'dtw_dist'};

n_di = length(disttypes);
%d_cl_results(1:n_di,1) = disttypes;
%d_cl_results(1:n_di,2) = {'1vsall'};
%d_cl_results(n_di+1:2*n_di,1) = disttypes;
%d_cl_results(n_di+1:2*n_di,2) = {'1vs1'};

for da= 1: n_da
    for di= 1: n_di
        dataset = char(datasets(da));
        disttype = char(disttypes(di));
        result_folder = strcat('/scratch/dy1n16/wSMDSEDMResults/',dataset);
    %     result_folder = strcat('Results_extended\',dataset);
        if exist(result_folder,'dir')~=7
            mkdir(result_folder) ;
        end
        data_folder = strcat(path,'/',dataset);
         if exist(data_folder,'dir')~=7
            error('The path is not correct.')
        end
        addpath(result_folder);
        addpath(data_folder);
    
        
        filename_distance_train = strcat(dataset,'_',disttype,...
                '_train.dsv');
        filename_distance_test = strcat(dataset,'_',disttype,...
                '_test.dsv');

        [~, target_train_ori, ~ ] = distance_dsv2data(filename_distance_train);
        k = min(min(hist(target_train_ori , unique(target_train_ori))),5);
        
        if exist(strcat(result_folder, '/', dataset,'_k-fold_indices_k_',num2str(k),'.csv'),'file') ~= 2
            indices = crossvalind('Kfold',target_train_ori,k);
            indicesfile_name_5 = strcat(dataset,'_k-fold_indices_k_',num2str(k),'.csv');
            csvwrite(strcat(result_folder, '/', dataset,'_k-fold_indices_k_',num2str(k),'.csv'),indices);
        else 
            indicesfile_name_5 = strcat(dataset,'_k-fold_indices_k_',num2str(k),'.csv');
        end
        
        
        errorrate_min = 1;
        al_op = 0.1;
        i=1;j=1
        for al = 0.9:-0.1:0.1
            i
            j=1;
            for q = -2.5 : 0.5: -1

                [errorrate, ~, ~, ~, ~, ~, ~,std_acc]  = ...
                    smds_main(filename_distance_train, 5, al, q, ...
                    disttype, '1vsall', 'distance', 'k-fold',...
                    indicesfile_name_5);
                eval(strcat(dataset,'_',disttype,'_','1vsall',...
                    '_k_fold_grid(i,j) = errorrate;'));
                j = j+1;
                if errorrate <= errorrate_min
                    errorrate_min = errorrate;
                    al_op = al;
                end
            end
            i = i+1;
        end
        
        i=1;j=1
        for al = al_op
            i
            j=1;
            for q = -2.5 : 0.1: -1

                [errorrate, ~, ~, ~, ~, ~, ~,~]  = ...
                    smds_main(filename_distance_train, 5, al, q, ...
                    disttype, '1vsall', 'distance', 'k-fold',...
                    indicesfile_name_5);
                eval(strcat(dataset,'_',disttype,'_','1vsall',...
                    '_k_fold(i,j) = errorrate;'));
                eval(strcat(dataset,'_',disttype,'_','1vsall',...
                    '_k_fold_std(i,j) = std_acc;'));
                j = j+1;
                if errorrate < errorrate_min
                    errorrate_min = errorrate;
                    al_op = al;
                end
            end
            i = i+1;
        end
        
        
        
        i=1;j=1
        for al = al_op
            i
            j=1;
            for q = -2.5 : 0.1: -1

                [errorrate, ~, Out_te, ~, ~, ~, target_test, ~]  = ...
                    smds_main(filename_distance_train, 5, al, q, ...
                    disttype, '1vsall', 'distance', 'test',...
                    filename_distance_test);
                eval(strcat(dataset,'_',disttype,'_','1vsall',...
                    '_test(i,j) = errorrate;'));
                j = j+1;
                
            end
            i = i+1;
        end
        
   
        

       


          

           

            save(strcat(result_folder,'/',dataset,'_',disttype,'_','1vsall','.mat'), ...
                strcat(dataset,'_',disttype,'_','1vsall','_test'),...
                strcat(dataset,'_',disttype,'_','1vsall','_k_fold'),...
                strcat(dataset,'_',disttype,'_','1vsall','_k_fold_grid'),...
                strcat(dataset,'_',disttype,'_','1vsall','_k_fold_std'));       

    %         save(strcat(result_folder,'\',dataset,'_',disttype,'_',cla_method,'_extended','.mat'), ...
    %             strcat(dataset,'_',disttype,'_',cla_method,'_selftest'),...
    %             strcat(dataset,'_',disttype,'_',cla_method,'_selftest_F_crits'),...
    %             strcat(dataset,'_',disttype,'_',cla_method,'_test'),...
    %             strcat(dataset,'_',disttype,'_',cla_method,'_test_F_crits'),...
    %             strcat(dataset,'_',disttype,'_',cla_method,'_k_fold'));

        


        proceed = da;
        
    end

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

function [data_ori, target_ori,  disttype_name ] = distance_dsv2data(filename)

disttype={'euclidean','squaredeuclidean',...
    'seuclidean','cityblock','minkowski','chebychev','correlation','cosine',...
    'hamming','jaccard','spearman','dtw','DTWR1','DTWR02','DTWR05','DTWR08',...
    'DDTWR1','DDTWR02','DDTWR05','DDTWR08','LCSS','ERP','MSM','TWE','WDTWG08',...
    'TWE','WDTWG0','WDTWG02','WDTWG05','WDDTWG0','WDDTWG02','WDDTWG05','WDDTWG08','ED'};
delimiters_line = strfind(filename,'_');
if ~ismember(filename(delimiters_line(1)+1:delimiters_line(2)-1),disttype)
    error('The format of distance filename can not be recognized')
else
    disttype_name = filename(delimiters_line(1)+1:delimiters_line(2)-1);
end

mat_ori = dlmread(filename);
data_ori = mat_ori(:,2:end);
target_ori = mat_ori(:,1);

end