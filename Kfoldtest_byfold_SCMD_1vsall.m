function [ Accuracy, model, Out_te, target_hat_folds ,Accuracy_fi] = Kfoldtest_byfold_SCMD_1vsall( k, data, target, data_test_ori, target_test, dim, alpha, q, disttype, indices, varargin )
% 
% This code is to performe K-fold cross-validation 
%
% INPUTS:
%   k       :  Parameter of K-fold                              [required]
%
%   data    :  Matrix of predictor data, where each row is one observation,
%              and each column is one predictor                 [required]
%
%   target  :  Array of class labels with each row corresponding
%              to the value of the corresponding row in X       [required]
%
%   dim     :  Dimension of configuration constructed by SMDS   [required]
%
%   alpha   :  Parameter of penalty term in SMDS model          [required]
%
%   q       :  Power of weight                                  [required]
%
%   disttype:  Type of distance
%
%   par_extra  : For noise, bounds and ancher setting in SMDS   [optional]
%% 

% indices         =crossvalind('Kfold',target,k);

fid = fopen('Output.txt','a+');
fprintf(fid,'--------------------------------------------------------------\n\n');
fprintf(fid,'                  Cross-validation of SCMD                    \n\n');
fprintf(fid,'--------------------------------------------------------------\n');
fprintf(fid,'--------------------------------------------------------------\n');
fprintf(fid,'Kfold: k=%d \n',k);
fprintf(fid,'--------------------------------------------------------------\n\n');
fclose(fid);   
Accuracy = zeros(1,k);
Out_te=struct;
target_hat_folds = nan(size(target_test,1),k);
for i=1:k
    fid = fopen('Output.txt','a+');
    fprintf(fid,'\n\n--------------------------------------------------------------\n');
    fprintf(fid,'                     %dth fold                            \n',i);
    fprintf(fid,'--------------------------------------------------------------\n\n');
    fclose(fid);
    test            = (indices == i);
    train           = ~test;
    if size(data,1) ~= size(data,2)     % when columns of 'data' represent features
        data_train      = data(train,:);
        target_train    = target(train,:);
        
    else                                % when 'data' is distance matrix
        data_train      = data(train,train);
        target_train    = target(train,:);
        data_test       = data_test_ori(:,train);
    end
%     time=tic;
if  ~ismember('distance',varargin)
    if ismember('extrapar',varargin)
     par_extra = cell2struct(varargin(find(ismember(varargin,'extrapar'),1)+1));
     [model,cutpoint] = SMDSTraining_1vsall(data_train, target_train, ...
         data_test ,target_test, dim, alpha,q, disttype,'extrapar', par_extra);
    else 
     [model,cutpoint] = SMDSTraining_1vsall(data_train, target_train, ...
         data_test ,target_test, dim, alpha, q, disttype);
    end
else
    if ismember('extrapar',varargin)
     par_extra = cell2struct(varargin(find(ismember(varargin,'extrapar'),1)+1));
     [model,cutpoint] = SMDSTraining_1vsall(data_train, target_train, ...
         data_test ,target_test, dim, alpha,q, disttype,'distance','extrapar', par_extra);
    else 
     [model,cutpoint] = SMDSTraining_1vsall(data_train, target_train, ...
         data_test ,target_test, dim, alpha, q, disttype,'distance');
    end
end
%     toc(time)
%     fid = fopen('Output.txt','a+');
%     fprintf(fid,'--------------------------------------------------------------\n');
%     fprintf(fid,'            Overall Model Building Time: %1.3fsec            \n',time);
%     fprintf(fid,'--------------------------------------------------------------\n\n\n');
%     fclose(fid);
    
    Out_te(i).Out_te=SQREDM_SMDS_test_1vsall(model,cutpoint);
    target_hat_folds(:,i) = Out_te(i).Out_te.PY_te_hat_fi;
    Accuracy(i)=Out_te(i).Out_te.Accuracy;
end
    Accuracy_fi = length(find(mode(target_hat_folds,2)==target_test))/size(target_test,1);
end