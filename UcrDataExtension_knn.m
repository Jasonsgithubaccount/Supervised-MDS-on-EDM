function [target_train_resam,data_train_resam] = UcrDataExtension_knn(datasets, path, varargin)


n_da = length(datasets);
varargin_char = {};
for i = 1 : length(varargin)
    if isnumeric(varargin{i})
        varargin_char{i} = mat2str(varargin{i});
    else
        varargin_char{i} = varargin{i};
    end
end

if  ismember('K',varargin_char)
    k = cell2mat(varargin(find(ismember(varargin_char,'K'),1)+1));
else
    k = 5; % default K
end

if  ismember('Distance',varargin_char)
    distance = char(varargin(find(ismember(varargin_char,'Distance'),1)+1));
else
    distance = 'euclidean'; % default distance type
end
for d= 1: n_da
    
    dataset = char(datasets(d));
    result_folder = strcat('UcrExtendedData\',dataset);
%     result_folder = strcat('Results_extended\',dataset);
    if exist(result_folder,'dir')~=7
        mkdir(result_folder) ;
    end
    
    data_folder = strcat(path,'\',dataset);
     if exist(data_folder,'dir')~=7
        error('The path is not correct.')
    end
    
    filepath_name_train = strcat(path,'\',dataset,'\',dataset,'_TRAIN.tsv');
    filepath_name_test = strcat(path,'\',dataset,'\',dataset,'_TEST.tsv');
    [data_train_ori, target_train_ori] = tsv2data(filepath_name_train);
    [data_train_resam, target_train_resam] = resampling_knn(data_train_ori, target_train_ori, 'K',k ,'Distance',distance);
    
    csvwrite(strcat('UcrExtendedData', '\', dataset,'\', dataset,'Knn',num2str(k),'Dist',distance,'_TRAIN.esv'),[target_train_resam,data_train_resam]);
    copyfile (filepath_name_test, strcat('UcrExtendedData','\',dataset,'\',dataset,'_TEST.tsv'));

end