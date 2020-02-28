function proceed=classificationCombined_results_forSimulateddata(datasets,k, path)
n_da = length(datasets);
addpath(path);
for d= 1: n_da
    dataset = char(datasets(d));
    trainfilename = strcat(dataset,'_TRAIN.ssv');
    testfilename = strcat(dataset,'_TEST.ssv');
    [ ~, ~, ~, ~, ~] = classificationCombined_main(trainfilename, k, 'test',testfilename);
      
end
proceed = d;
end