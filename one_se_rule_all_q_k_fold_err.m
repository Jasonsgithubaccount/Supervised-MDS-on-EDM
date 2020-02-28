function one_se_all_q =one_se_rule_all_q_k_fold_err(k_fold_err,k_fold_std,test_err,k)

one_se_all_q = nan(size(k_fold_err,1),5);
for i = 1:size(k_fold_err,1)
[test_one_se_exa_min,test_one_se_exa_max,q_min,q_max,min_k_fold_err, min_k_fold_std_min, min_k_fold_std_max, q_min_minacc, q_max_minacc, k_fold_one_se_exa_min, k_fold_one_se_exa_max]=one_se_rule_one_dim_k_fold_err(k_fold_err(i,:),k_fold_std(i,:),test_err(i,:),k);
one_se_all_q(i,1) = test_one_se_exa_min;
one_se_all_q(i,2) = test_one_se_exa_max;
one_se_all_q(i,3) = q_min;
one_se_all_q(i,4) = q_max;
one_se_all_q(i,5) = min_k_fold_err;
one_se_all_q(i,6) = min_k_fold_std_min;
one_se_all_q(i,7) = min_k_fold_std_max;
one_se_all_q(i,8) = q_min_minacc;
one_se_all_q(i,9) = q_max_minacc;
one_se_all_q(i,10) = k_fold_one_se_exa_min;
one_se_all_q(i,11) = k_fold_one_se_exa_max;
end

end




function [test_one_se_exa_min,test_one_se_exa_max,q_min,q_max,min_k_fold_err, min_k_fold_std_min, min_k_fold_std_max, q_min_minacc, q_max_minacc, k_fold_one_se_exa_min, k_fold_one_se_exa_max]=one_se_rule_one_dim_k_fold_err(k_fold_err,k_fold_std,test_err,k)

min_k_fold_err = min(k_fold_err);
col_num = find(k_fold_err(:)==min_k_fold_err);
min_k_fold_std_min=k_fold_std(min(col_num));
min_k_fold_std_max=k_fold_std(max(col_num));
k_fold_one_se_exa_min = min_k_fold_err+min_k_fold_std_min/sqrt(k);
k_fold_one_se_exa_max = min_k_fold_err+min_k_fold_std_max/sqrt(k);
col_num_one_se_min = min(find(k_fold_err(1:min(col_num))<=k_fold_one_se_exa_min));
col_num_one_se_max = max(find(k_fold_err(max(col_num)+1:end)<=k_fold_one_se_exa_max))+max(col_num);
if isempty(col_num_one_se_max)
    col_num_one_se_max = length(k_fold_err);
end
test_one_se_exa_min = test_err(col_num_one_se_min);
test_one_se_exa_max = test_err(col_num_one_se_max);
q_min = -2.6+0.1*col_num_one_se_min;
q_max = -2.6+0.1*col_num_one_se_max;
q_min_minacc = -2.6+0.1*min(col_num);
q_max_minacc = -2.6+0.1*max(col_num);
end