function [DataFile_name] = SimulatedExamples(problemname, n, dim, p)
%
% problemname:  {'Constant','Two-sided','Linear'} 
% collected from existing literature
%
% noisetype = 'additive', 'multiplicative', 'log-normal'
% n: number of anchors and sensors


%% Load Data

rng('default');
randstate = rng('shuffle');


    
switch problemname        
    case 'Constant'
        PY      = [ones(floor(n/2), dim); -ones(n - floor(n/2), dim)];
        Y       = [ones(floor(n/2),1); 2*ones(n - floor(n/2), 1)];
        PS      = randn(n,dim);
        PS      = PY*0.4 + PS;
        PP      = [Y PS];
        indices = crossvalind('Kfold',Y,10);
        PP_train = PP(find(indices>p),:);
        PP_test = PP(find(indices<=p),:);
        
        if exist('SimulatedTESTData','dir')~=7
            mkdir('SimulatedTESTData') ;
        end
        csvwrite(strcat('SimulatedTESTData', '\', problemname,'N',num2str(n),'Dim',num2str(dim),'Percent',num2str(p*10),'_TRAIN.ssv'),PP_train);
        csvwrite(strcat('SimulatedTESTData', '\', problemname,'N',num2str(n),'Dim',num2str(dim),'Percent',num2str(p*10),'_TEST.ssv'),PP_test);
        DataFile_name = strcat(problemname,'N',num2str(n),'Dim',num2str(dim),'Percent',num2str(p*10),'.ssv');
        figure;scatter(PP_train(:,2),PP_train(:,3),[],PP_train(:,1));
        title(strcat(problemname,': N: ',num2str(n),' Dim: ',num2str(dim),' Percent: ',num2str(p*10),', Train'));
        saveas(gcf, strcat('SimulatedTESTData', '\',problemname,'_N_',num2str(n),'_Dim_',num2str(dim),'_Percent_',num2str(p*10),'_Train','.png'));
        close;
        figure;scatter(PP_test(:,2),PP_test(:,3),[],PP_test(:,1));
        title(strcat(problemname,': N: ',num2str(n),' Dim: ',num2str(dim),' Percent: ',num2str(p*10),', Test'));
        saveas(gcf, strcat('SimulatedTESTData', '\',problemname,'_N_',num2str(n),'_Dim_',num2str(dim),'_Percent_',num2str(p*10),'_Test','.png'));
        close;
    case 'Twosided'
        PY      = [ones(floor(n/4), dim); -ones(floor(n/2) - floor(n/4), dim); zeros(n - floor(n/2), dim)];
        Y       = [ones(floor(n/4), 1); -ones(floor(n/2) - floor(n/4), 1); zeros(n - floor(n/2), 1)];
        PS      = randn(n,dim);
        PS      = PY + PS;
        Y(Y==-1)=1;
        Y(Y==0)=2;
        PP      = [Y PS];
        indices = crossvalind('Kfold',Y,10);
        PP_train = PP(find(indices>p),:);
        PP_test = PP(find(indices<=p),:);
        
        if exist('SimulatedTESTData','dir')~=7
            mkdir('SimulatedTESTData') ;
        end
        csvwrite(strcat('SimulatedTESTData', '\', problemname,'N',num2str(n),'Dim',num2str(dim),'Percent',num2str(p*10),'_TRAIN.ssv'),PP_train);
        csvwrite(strcat('SimulatedTESTData', '\', problemname,'N',num2str(n),'Dim',num2str(dim),'Percent',num2str(p*10),'_TEST.ssv'),PP_test);
        DataFile_name = strcat(problemname,'N',num2str(n),'Dim',num2str(dim),'Percent',num2str(p*10),'.ssv');
        figure;scatter(PP_train(:,2),PP_train(:,3),[],PP_train(:,1));
        title(strcat(problemname,': N: ',num2str(n),' Dim: ',num2str(dim),' Percent: ',num2str(p*10),', Train'));
        saveas(gcf, strcat('SimulatedTESTData', '\',problemname,'_N_',num2str(n),'_Dim_',num2str(dim),'_Percent_',num2str(p*10),'_Train','.png'));
        close;
        figure;scatter(PP_test(:,2),PP_test(:,3),[],PP_test(:,1));
        title(strcat(problemname,': N: ',num2str(n),' Dim: ',num2str(dim),' Percent: ',num2str(p*10),', Test'));
        saveas(gcf, strcat('SimulatedTESTData', '\',problemname,'_N_',num2str(n),'_Dim_',num2str(dim),'_Percent_',num2str(p*10),'_Test','.png'));
        close;
    case 'TwosidedBound'
        PY      = [ones(floor(n/4), dim); -ones(floor(n/2) - floor(n/4), dim); zeros(n - floor(n/2), dim)];
        Y       = [ones(floor(n/4), 1); -ones(floor(n/2) - floor(n/4), 1); zeros(n - floor(n/2), 1)];
        PS      = randn(n,dim);
        PS      = PY + PS;
        Y(Y==-1)=1;
        Y(Y==0)=2;
        PP      = [Y PS];
%         indices = crossvalind('Kfold',Y,10);
        PP_train = PP;
        rmax = ceil(max(max(PS)));
        rmin = floor(min(min(PS)));
        [X,Y]=meshgrid(rmin:0.1:rmax,rmin:0.1:rmax);
        X_1 = X(:);
        Y_1 = Y(:);
        PP_test = [ones(size(X_1,1),1) X_1 Y_1];
        if exist('SimulatedTESTData','dir')~=7
            mkdir('SimulatedTESTData') ;
        end
        csvwrite(strcat('SimulatedTESTData', '\', problemname,'N',num2str(n),'Dim',num2str(dim),'_TRAIN.ssv'),PP_train);
        csvwrite(strcat('SimulatedTESTData', '\', problemname,'N',num2str(n),'Dim',num2str(dim),'_TEST.ssv'),PP_test);
        DataFile_name = strcat(problemname,'N',num2str(n),'Dim',num2str(dim),'.ssv');
        figure;scatter(PP_train(:,2),PP_train(:,3),[],PP_train(:,1));
        title(strcat(problemname,': N: ',num2str(n),' Dim: ',num2str(dim),', Train'));
        saveas(gcf, strcat('SimulatedTESTData', '\',problemname,'_N_',num2str(n),'_Dim_',num2str(dim),'_Train','.png'));
        close;
        figure;scatter(PP_test(:,2),PP_test(:,3),[],PP_test(:,1));
        title(strcat(problemname,': N: ',num2str(n),' Dim: ',num2str(dim),', Test'));
        saveas(gcf, strcat('SimulatedTESTData', '\',problemname,'_N_',num2str(n),'_Dim_',num2str(dim),'_Test','.png'));
        close;
    case 'Linear'
        PY      = ones(n, dim).*(1:n)'/n;
        Y       = [ones(floor(n/2),1); 2*ones(n - floor(n/2), 1)];
        PS      = randn(n,dim);
        PS      = PY + PS;
        PP      = [Y PS];
        indices = crossvalind('Kfold',Y,10);
        PP_train = PP(find(indices>p),:);
        PP_test = PP(find(indices<=p),:);
        
        if exist('SimulatedTESTData','dir')~=7
            mkdir('SimulatedTESTData') ;
        end
        csvwrite(strcat('SimulatedTESTData', '\', problemname,'N',num2str(n),'Dim',num2str(dim),'Percent',num2str(p*10),'_TRAIN.ssv'),PP_train);
        csvwrite(strcat('SimulatedTESTData', '\', problemname,'N',num2str(n),'Dim',num2str(dim),'Percent',num2str(p*10),'_TEST.ssv'),PP_test);
        DataFile_name = strcat(problemname,'N',num2str(n),'Dim',num2str(dim),'Percent',num2str(p*10),'.ssv');
        figure;scatter(PP_train(:,2),PP_train(:,3),[],PP_train(:,1));
        title(strcat(problemname,': N: ',num2str(n),' Dim: ',num2str(dim),' Percent: ',num2str(p*10),', Train'));
        saveas(gcf, strcat('SimulatedTESTData', '\',problemname,'_N_',num2str(n),'_Dim_',num2str(dim),'_Percent_',num2str(p*10),'_Train','.png'));
        close;
        figure;scatter(PP_test(:,2),PP_test(:,3),[],PP_test(:,1));
        title(strcat(problemname,': N: ',num2str(n),' Dim: ',num2str(dim),' Percent: ',num2str(p*10),', Test'));
        saveas(gcf, strcat('SimulatedTESTData', '\',problemname,'_N_',num2str(n),'_Dim_',num2str(dim),'_Percent_',num2str(p*10),'_Test','.png'));
        close;
        
     case 'Hastie'
        rng default % For reproducibility
        n_expand = 10;
        grnpop  = mvnrnd([1,0],eye(2),n);
        redpop  = mvnrnd([0,1],eye(2),n);
        redpts  = zeros(n_expand*n,2);grnpts = redpts;
        for i = 1:n_expand*n
            grnpts(i,:) = mvnrnd(grnpop(randi(n_expand),:),eye(2)*0.02);
            redpts(i,:) = mvnrnd(redpop(randi(n_expand),:),eye(2)*0.02);
        end
        PS = [grnpts;redpts];
        Y = ones(2*n_expand*n,1);
        Y(n_expand*n+1:2*n_expand*n) = 2;
        PP      = [Y PS];
        indices = crossvalind('Kfold',Y,10);
        PP_train = PP(find(indices>p),:);
        PP_test = PP(find(indices<=p),:);
        
        if exist('SimulatedTESTData','dir')~=7
            mkdir('SimulatedTESTData') ;
        end
        csvwrite(strcat('SimulatedTESTData', '\', problemname,'N',num2str(n),'Dim',num2str(dim),'Percent',num2str(p*10),'_TRAIN.ssv'),PP_train);
        csvwrite(strcat('SimulatedTESTData', '\', problemname,'N',num2str(n),'Dim',num2str(dim),'Percent',num2str(p*10),'_TEST.ssv'),PP_test);
        DataFile_name = strcat(problemname,'N',num2str(n),'Dim',num2str(dim),'Percent',num2str(p*10),'.ssv');
        figure;scatter(PP_train(:,2),PP_train(:,3),[],PP_train(:,1));
        title(strcat(problemname,': N: ',num2str(n),' Dim: ',num2str(dim),' Percent: ',num2str(p*10),', Train'));
        saveas(gcf, strcat('SimulatedTESTData', '\',problemname,'_N_',num2str(n),'_Dim_',num2str(dim),'_Percent_',num2str(p*10),'_Train','.png'));
        close;
        figure;scatter(PP_test(:,2),PP_test(:,3),[],PP_test(:,1));
        title(strcat(problemname,': N: ',num2str(n),' Dim: ',num2str(dim),' Percent: ',num2str(p*10),', Test'));
        saveas(gcf, strcat('SimulatedTESTData', '\',problemname,'_N_',num2str(n),'_Dim_',num2str(dim),'_Percent_',num2str(p*10),'_Test','.png'));
        close;
        
    case 'Circle'
        rng(1); % For reproducibility
        r       = sqrt(rand(floor(n/2),1)); % Radius
        t       = 2*pi*rand(floor(n/2),1);  % Angle
        data1   = [r.*cos(t), r.*sin(t)]; % Points
        r2      = sqrt(3*rand(n-floor(n/2),1)+1); % Radius
        t2      = 2*pi*rand(n-floor(n/2),1);      % Angle
        data2   = [r2.*cos(t2), r2.*sin(t2)]; % points
        PS      = [data1;data2];
        Y       = [ones(floor(n/2),1); 2*ones(n - floor(n/2), 1)];
        PP      = [Y PS];
        indices = crossvalind('Kfold',Y,10);
        PP_train = PP(find(indices>p),:);
        PP_test = PP(find(indices<=p),:);
        
        if exist('SimulatedTESTData','dir')~=7
            mkdir('SimulatedTESTData') ;
        end
        csvwrite(strcat('SimulatedTESTData', '\', problemname,'N',num2str(n),'Dim',num2str(dim),'Percent',num2str(p*10),'_TRAIN.ssv'),PP_train);
        csvwrite(strcat('SimulatedTESTData', '\', problemname,'N',num2str(n),'Dim',num2str(dim),'Percent',num2str(p*10),'_TEST.ssv'),PP_test);
        DataFile_name = strcat(problemname,'N',num2str(n),'Dim',num2str(dim),'Percent',num2str(p*10),'.ssv');
        figure;scatter(PP_train(:,2),PP_train(:,3),[],PP_train(:,1));
        title(strcat(problemname,': N: ',num2str(n),' Dim: ',num2str(dim),' Percent: ',num2str(p*10),', Train'));
        saveas(gcf, strcat('SimulatedTESTData', '\',problemname,'_N_',num2str(n),'_Dim_',num2str(dim),'_Percent_',num2str(p*10),'_Train','.png'));
        close;
        figure;scatter(PP_test(:,2),PP_test(:,3),[],PP_test(:,1));
        title(strcat(problemname,': N: ',num2str(n),' Dim: ',num2str(dim),' Percent: ',num2str(p*10),', Test'));
        saveas(gcf, strcat('SimulatedTESTData', '\',problemname,'_N_',num2str(n),'_Dim_',num2str(dim),'_Percent_',num2str(p*10),'_Test','.png'));
        close;
        
    otherwise
        disp('input a problen name');
        
end

end



