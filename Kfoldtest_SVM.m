function [ errorrate ] = Kfoldtest_SVM( k, data, target, kernel )
%KFOLDTEST Summary of this function goes here
%   Detailed explanation goes here


indices         =crossvalind('Kfold',target,k);
fclose('all');
if exist('Output.txt') ==2; delete('Output.txt'); end
fid             =fopen('Output.txt','a+');
    fprintf(fid,'--------------------------------------------------------------\n\n');
    fprintf(fid,'                  Cross-validation of SVM                    \n\n');
    fprintf(fid,'--------------------------------------------------------------\n');
    fprintf(fid,'--------------------------------------------------------------\n');
    fprintf(fid,'Kfold: k=%d \n',k);
    fprintf(fid,'Parameters of SCMD : kernel   = %s;\n', kernel);
    %fprintf(fid,'                     alpha = %1.3f;\n', alpha);
    %fprintf(fid,'                     q     = %1.3f.\n',  q);
    fprintf(fid,'--------------------------------------------------------------\n\n');
    
for i=1:k
    fprintf(fid,'\n\n--------------------------------------------------------------\n');
    fprintf(fid,'                     %dth fold                            \n',i);
    fprintf(fid,'--------------------------------------------------------------\n\n');
    test            = (indices == i);
    train           = ~test;
    data_train      = data(train,:);
    target_train    = target(train,:);
    data_test       = data(test,:);
    target_test     = target(test,:);
    tref= tic;
    model = SVMTraining(data_train, target_train, kernel,fid);
    time= toc(tref); 
    fprintf(fid,'--------------------------------------------------------------\n');
    fprintf(fid,'            Overall Model Building Time: %1.3fsec            \n',time);
    fprintf(fid,'--------------------------------------------------------------\n\n\n');
    Out_te = SVMTesting(model,data_test,target_test,fid);
    errorrate(i)=Out_te.Errorrate;
end
fprintf(fid,'\n\n--------------------------------------------------------------\n');
fprintf(fid,'                  Overall Accuracy Rate: %3d                  \n',mean(errorrate));
fprintf(fid,'--------------------------------------------------------------\n\n\n');
fprintf('Overall Accuracy Rate: %3d  \n',mean(errorrate));
fclose('all');
movefile('Output.txt',strcat('Linear_3c_p_10_s_500','_SVD_','_kernel_',kernel,'_recall_',num2str(mean(errorrate),'%1.2f'),'.txt'));
end
