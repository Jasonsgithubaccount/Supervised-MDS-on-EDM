% case 'Dermatology'
x=dlmread('dermatology.data',',');
Dermatology=[x(:,end),x(:,1:end-2)];
Y = Dermatology(:,1);

for index = 1 : 30
    indices = crossvalind('Kfold',Y,3);
    PP_train = Dermatology(find(indices<3),:);
    PP_test = Dermatology(find(indices==3),:);
    if exist(strcat('UCISampleSets\DermatologyTR',num2str(size(PP_train,1)),'Index',num2str(index)),'dir')~=7
            mkdir(strcat('UCISampleSets\DermatologyTR',num2str(size(PP_train,1)),'Index',num2str(index))) ;
    end
    csvwrite(strcat('UCISampleSets\DermatologyTR',num2str(size(PP_train,1)),'Index',num2str(index), '\DermatologyTR',num2str(size(PP_train,1)),'Index',num2str(index),'_TRAIN.ssv'),PP_train);
    csvwrite(strcat('UCISampleSets\DermatologyTR',num2str(size(PP_train,1)),'Index',num2str(index), '\DermatologyTR',num2str(size(PP_train,1)),'Index',num2str(index),'_TEST.ssv'),PP_test);
end


% case 'Skin'
x=dlmread('Skin_NonSkin.txt',',');
Skin=[x(:,end),x(:,1:end-2)];
Y = Skin(:,1);

for index = 1 : 30
    indices = crossvalind('Kfold',Y,3);
    PP_train = Skin(find(indices<3),:);
    PP_test = Skin(find(indices==3),:);
    if exist(strcat('UCISampleSets\SkinTR',num2str(size(PP_train,1)),'Index',num2str(index)),'dir')~=7
            mkdir(strcat('UCISampleSets\SkinTR',num2str(size(PP_train,1)),'Index',num2str(index))) ;
    end
    csvwrite(strcat('UCISampleSets\SkinTR',num2str(size(PP_train,1)),'Index',num2str(index), '\SkinTR',num2str(size(PP_train,1)),'Index',num2str(index),'_TRAIN.ssv'),PP_train);
    csvwrite(strcat('UCISampleSets\SkinTR',num2str(size(PP_train,1)),'Index',num2str(index), '\SkinTR',num2str(size(PP_train,1)),'Index',num2str(index),'_TEST.ssv'),PP_test);
end


% case 'Skin'
x=dlmread('wine.data',',');
x=x(:,1:end-1);
Wine=x(:,1:end);
Y = x(:,1);

for index = 1 : 30
    indices = crossvalind('Kfold',Y,3);
    PP_train = Wine(find(indices<3),:);
    PP_test = Wine(find(indices==3),:);
    if exist(strcat('UCISampleSets\WineTR',num2str(size(PP_train,1)),'Index',num2str(index)),'dir')~=7
            mkdir(strcat('UCISampleSets\WineTR',num2str(size(PP_train,1)),'Index',num2str(index))) ;
    end
    csvwrite(strcat('UCISampleSets\WineTR',num2str(size(PP_train,1)),'Index',num2str(index), '\WineTR',num2str(size(PP_train,1)),'Index',num2str(index),'_TRAIN.ssv'),PP_train);
    csvwrite(strcat('UCISampleSets\WineTR',num2str(size(PP_train,1)),'Index',num2str(index), '\WineTR',num2str(size(PP_train,1)),'Index',num2str(index),'_TEST.ssv'),PP_test);
end



% case 'Dermatology'
x=dlmread('dermatology.data',',');
Dermatology=[x(:,end),x(:,1:end-2)];
Y = Dermatology(:,1);

for index = 1 : 30
    indices = crossvalind('Kfold',Y,3);
    PP_train = Dermatology(find(indices<3),:);
    PP_test = Dermatology(find(indices==3),:);
    if exist(strcat('UCISampleSets\DermatologyTR',num2str(size(PP_train,1)),'Index',num2str(index)),'dir')~=7
            mkdir(strcat('UCISampleSets\DermatologyTR',num2str(size(PP_train,1)),'Index',num2str(index))) ;
    end
    csvwrite(strcat('UCISampleSets\DermatologyTR',num2str(size(PP_train,1)),'Index',num2str(index), '\DermatologyTR',num2str(size(PP_train,1)),'Index',num2str(index),'_TRAIN.ssv'),PP_train);
    csvwrite(strcat('UCISampleSets\DermatologyTR',num2str(size(PP_train,1)),'Index',num2str(index), '\DermatologyTR',num2str(size(PP_train,1)),'Index',num2str(index),'_TEST.ssv'),PP_test);
end