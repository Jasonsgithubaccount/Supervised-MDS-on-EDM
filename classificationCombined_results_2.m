function proceed=classificationCombined_results_2(datasets,k, path)
n_da = length(datasets);
addpath(path);
for d= 1: n_da
    dataset = char(datasets(d));
    trainfilename = strcat(path, '\', dataset, '\', dataset, '_TRAIN.tsv');
    testfilename = strcat(path, '\', dataset, '\', dataset, '_TEST.tsv');
    [ ~, ~, ~, ~, ~] = classificationCombined_main(trainfilename, k, 'test',testfilename);
      
end
proceed = d;
end