function errorrate = smds_main_errorrate_kfold_distance(filename_train, dim, alpha, q, disttype,cla_method,indicesfile_name )
[errorrate,~, ~, ~, ~, ~, ~] = smds_main(filename_train, dim, alpha, q, disttype,cla_method,'distance','k-fold', indicesfile_name);
end