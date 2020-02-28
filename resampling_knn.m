function [data_resam, target_resam] = resampling_knn(data, target, varargin)
varargin_char = {};
for i = 1 : length(varargin)
varargin_char{i} = mat2str(varargin{i});
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
%initialize output and writing target as an empty matrix
data_resam = zeros(0, size(data,2));
target_resam = zeros(0, 1);
unitarget = unique(target);
nunitarget = length(unitarget);

for nidx = 1 : nunitarget
    idx_unitarget = find(unitarget(nidx)==target);
    data_idx = data(idx_unitarget,:);
    [kIdxs,D] = knnsearch(data_idx,data_idx,'K',k+1,'Distance', distance);
    for i = 1 : length(idx_unitarget)
        
        for j = 1 : k +1
            delta = rand(1);
            data_iden = data_idx(i,:) - delta*(data_idx(kIdxs(i,j),:) - data_idx(i,:));
            data_resam = [data_resam ; data_iden];
            target_resam = [target_resam; unitarget(nidx)];
        end
    end
end