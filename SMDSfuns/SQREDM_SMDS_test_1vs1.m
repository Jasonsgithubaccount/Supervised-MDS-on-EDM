function Out_te=SQREDM_SMDS_test_1vs1(model,cutpoint)
%This code is 
%% 



%if isfield(Out,'X')
%else error('Training result has not been constructed.')
%end

    fid = fopen('Output.txt','a+');
    T=length(model);
    n_te = size(model(1).D_te,1);
    dim = model(1).Out.dim;
    Out_te.PY_te_hat = zeros(n_te,T);
    Out_te.X_te = zeros(dim,n_te,T);
    Out_te.PY_te = model(1).pars_2c.PY_te';
    F_crit_1s   = zeros(n_te,T);
    F_crit_2s   = zeros(n_te,T);
    X_new_1s    = zeros(dim,n_te,T);
    X_new_2s    = zeros(dim,n_te,T);
   
    
    for t=1:T
        D_te = full(model(t).D_te);
        Out  = model(t).Out;
        alpha = Out.alpha;
        q = Out.q;
        pars = model(t).pars_2c;
        a           = model(t).lable_a;
        b           = model(t).lable_b;
        fprintf(fid,'\n--------------------------Testing-----------------------------\n');
        fprintf(fid,'        Class_a Label:%1d    Class_b Label:%1d      \n', a, b);
        fprintf(fid,'--------------------------------------------------------------\n');
       
        for i = 1:size(D_te,1)
    %         fprintf(fid,'Item %d: ',i);
    %         tref= tic;
            cf= TestSuperMDSSingleObs (D_te(i,:), alpha, Out, pars.PY,a, b,q);
    %         time= toc(tref); 
    %         fprintf(fid,'Time: %1.3fsec\n',time);
            F_crit_1s(i,t)   = cf.F_crit_1;
            F_crit_2s(i,t)   = cf.F_crit_2;
            %fprintf(fid,'Item %d: F2-F1: %1.3f; F2/F1: %1.3f; PY:%1d\n', i,F_crit_2s(i)-F_crit_1s(i),F_crit_2s(i)/F_crit_1s(i),Out_te.PY_te(i) );
            X_new_1s(:,i,t)  = cf.X_new_1;
            X_new_2s(:,i,t)  = cf.X_new_2;
        end
    
        
        
            %cutpoint    = 0;
            %Out_te.PY_te_hat = a*ones(size(D_te,1),T);
            %Out_te.PY_te_hat(:,t)=a;
            Out_te.PY_te_F_crit(:,t)=F_crit_1s(:,t)-F_crit_2s(:,t)-cutpoint(t);
            Out_te.PY_te_hat(:,t) = a;
            Out_te.PY_te_hat(Out_te.PY_te_F_crit(:,t) > 0,t)= b;
            
            
    end
    Out_te.PY_te_hat_fi = zeros(n_te,1);
    Out_te.PY_te_hat_fi = mode(Out_te.PY_te_hat,2);
    Out_te.X_new_1s = X_new_1s;
    Out_te.X_new_2s = X_new_2s;

    
    
    %Out_te.PY_te_F_crit(:,t)=F_crit_1s;
            
    
            
     
%             if ~ismember(a,Out_te.PY_te)
%                 Out_te.PY_te_F_crit(:,t) = -1;
%             end
%             Out_te.PY_te_hat(g_f,t) = b;
%             Out_te.PY_te_F_crit(g_f,t) = F_crit_1s(g_f)-F_crit_2s(g_f);
%             %Out_te.PY_te_F_crit(g_f,t)=F_crit_2s(g_f);
%             if ~ismember(b,Out_te.PY_te)
%                 Out_te.PY_te_F_crit(g_f,t) = -1;
%             end
%             %Out_te.X_te = zeros(size(Out.X,1),size(D_te,1),T);
%             Out_te.X_te(:,:,t) = X_new_1s;
%             Out_te.X_te(:,g_f,t) = X_new_2s(:,g_f);
%             %Out_te.Errorrate = (size(pars.PY_te,2)-length(find(Out_te.PY_te_hat~=pars.PY_te(1,:))))/size(pars.PY_te,2);
%         else 
%             
%         %figure;scatter3( Out_te.X_te(1,:),Out_te.X_te(2,:),Out_te.X_te(3,:),[],pars.PY_te(1,:));
%         %figure;scatter3( Out_te.X_te(1,:),Out_te.X_te(2,:),Out_te.X_te(3,:),[],Out_te.PY_te_hat(1,:));
%         %fprintf('Error Rate %3d',Out_te.Errorrate);
        
  

% 
%     % Out_te.PY_te_hat_fi = mode(Out_te.PY_te_hat,2);
    Out_te.Accuracy = (size(Out_te.PY_te_hat_fi,1)-length(find(Out_te.PY_te_hat_fi~=Out_te.PY_te)))/size(Out_te.PY_te_hat_fi,1);


    fprintf(fid,'\n--------------------------Testing----------------------------\n');
    fprintf(fid,'        Prediction Result of Each Model (Accuracy: %1.3f)      \n', Out_te.Accuracy);
    fprintf(fid,'--------------------------------------------------------------\n');
    for i = 1:n_te
        fprintf(fid,'Item %d: ',i);
        for t = 1 : T
            fprintf(fid,'%d  ',Out_te.PY_te_hat(i,t));
            fprintf(fid,'%d  ',Out_te.PY_te_F_crit(i,t));
        end
        fprintf(fid,', PY: %d PY_hat: %d ',Out_te.PY_te(i),Out_te.PY_te_hat_fi(i));
        fprintf(fid,'\n');
    end

    ClassLabel=unique(Out_te.PY_te);    
    k=length(ClassLabel);
    fprintf(fid,'--------------------------------------------------------------\n');
    fprintf(fid,'Precision Rates\n');
    for i=1:k
        Out_te.Precision(i,1) = ClassLabel(i);
        Out_te.Precision(i,2) = length(find(Out_te.PY_te_hat_fi==ClassLabel(i)...
            & Out_te.PY_te == Out_te.PY_te_hat_fi ))/ length(find(Out_te.PY_te_hat_fi==ClassLabel(i)));
        fprintf(fid,'%d %.2f%%; ',Out_te.Precision(i,:));
    end
    fprintf(fid,'\n--------------------------------------------------------------\n\n');
    fclose(fid);
end


function cf= TestSuperMDSSingleObs (D_te, alpha, Out, PY, a, b,q)
    
    n       = size(Out.X,2); 
    dim     = size(Out.X,1);
    itmax   = 1000;
    H = D_te;
    %q = 2 ;

%     H2 = exp(-H.^2/ std2(H)^2);
%     H2=H2_Gaussian(H,q);
%     H2    = Out.H2; 
    H2    = H.^q; 
    H2(H==0)=0;
    H2(H2==inf)=0;
    
    % Case 1: 
    %Y_te        = 1;
    g_a     = find(PY(1,:)==a);
    X_new_1     = mean( Out.X(1:dim,:),2);
    DY_mod      = ones(1,n);
    DY_mod(g_a) = 0;
    %H2 = (ones(size(DY_mod,1),1)*beta).^(-1).*H2;
    F_crit_1    = evaltestcrit(D_te, Out, X_new_1, DY_mod, alpha, H2);
    
    %DY_beta_mod = (ones(size(DY_mod,1),1)*beta).^(-1).*DY_mod;
    %F_crit_1    = evaltestcrit(D_te, Out, X_new_1, DY_beta_mod, alpha, H2);
    
    for iter= 1:itmax

        fn_1        = sqrt(sum(((Out.X-X_new_1*ones(1,size(Out.X,2))).^2),1));
        D_te_hat    = (2*(1-alpha)*D_te - (-2)*alpha*DY_mod.*D_te).*H2;
        X_new_1     = sum((ones(dim,1)*H2).*Out.X,2)/sum(H2,2) + (1/(2*sum(H2,2)*(1-alpha)))*sum((ones(dim,1)*D_te_hat).*(X_new_1*ones(1,n)-Out.X)./(ones(dim,1)*fn_1),2);
        F_crit_1o   = F_crit_1;
        F_crit_1    = evaltestcrit(D_te, Out, X_new_1, DY_mod, alpha, H2);
        %[~,FSum,FSqSum] = evaltestcrit(D_te, Out, X_new_1, DY_beta_mod, alpha, H2);
        ErrObj      = abs(F_crit_1o-F_crit_1)/abs(F_crit_1o+1);
%         if iter==1; 
%             fprintf(fid,'Iter: %d  F_crit_1o:%.3e F_crit_1:%.3e ErrObj:%.3e; ',iter, F_crit_1o, F_crit_1, ErrObj);
%         end
%         if ErrObj<sqrt(n)*1e-6 & iter>9; 
        if ErrObj<1e-6 & iter>9
            %fprintf(fid,'Iter: %d  F_crit_1o:%.3e F_crit_1:%.3e ErrObj:%.3e; ',iter, F_crit_1o, F_crit_1, ErrObj);
            %fprintf('FSum:%.3e; FSqSum:%.3e \n',FSum,FSqSum);
            break; 
        end 

    end
    
    % Case 2: 
    %Y_te        = -1;
    g_b     = find(PY(1,:)==b);
    X_new_2     = mean( Out.X(1:dim,:),2);
    DY_mod      = ones(1,n);
    DY_mod(g_b) = 0;
    %H2 = (ones(size(DY_mod,1),1)*beta).^(-1).*H2;
    F_crit_2    = evaltestcrit(D_te, Out, X_new_2, DY_mod, alpha, H2);
    %DY_beta_mod = (ones(size(DY_mod,1),1)*beta).^(-1).*DY_mod;
    %F_crit_2    = evaltestcrit(D_te, Out, X_new_2, DY_beta_mod, alpha, H2);
    for iter= 1:itmax

        fn_2        = sqrt(sum(((Out.X-X_new_2*ones(1,size(Out.X,2))).^2),1));
        D_te_hat    = (2*(1-alpha)*D_te - (-2)*alpha*DY_mod.*D_te).*H2;
        X_new_2     = sum((ones(dim,1)*H2).*Out.X,2)/sum(H2,2) + (1/(2*sum(H2,2)*(1-alpha)))*sum((ones(dim,1)*D_te_hat).*(X_new_2*ones(1,n)-Out.X)./(ones(dim,1)*fn_2),2);
        F_crit_2o   = F_crit_2;
        F_crit_2    = evaltestcrit(D_te, Out, X_new_2, DY_mod, alpha, H2);
        %[~,FSum,FSqSum] = evaltestcrit(D_te, Out, X_new_2, DY_beta_mod, alpha, H2);
        ErrObj      = abs(F_crit_2o-F_crit_2)/abs(F_crit_2o+1);
%         if iter==1; 
%             fprintf(fid,'Iter:%d  F_crit_2o: %.3e F_crit_2: %.3e ErrObj: %.3e ',iter, F_crit_2o, F_crit_2, ErrObj);
%         end
%         if ErrObj<sqrt(n)*1e-6 & iter>9; 
        if ErrObj<1e-6 & iter>9
%             fprintf(fid,'Iter:%d  F_crit_2o: %.3e F_crit_2: %.3e ErrObj: %.3e; ',iter, F_crit_2o, F_crit_2, ErrObj);
            %fprintf('FSum:%.3e; FSqSum:%.3e \n',FSum,FSqSum);
            break; 
        end 

    end

    cf.X_new_1      = X_new_1;
    cf.F_crit_1     = F_crit_1;
    cf.X_new_2      = X_new_2;
    cf.F_crit_2     = F_crit_2;
    
end

function [F_crit,FSum,FSqSum]= evaltestcrit(D_te, Out, X_new, DY_mod, alpha ,H2)
    
    n   = size(DY_mod,2); 
    %n_1 = size(PY(PY(1,:)==1));
    %n_2 = size(PY(PY(1,:)==-1));
    %t   = size(par.PS_te,2);
    % Case 1: y_new
    z_z_new = Out.X-X_new*ones(1,size(Out.X,2));
    multi_z_z_new_2 = (Out.X-X_new*ones(1,size(Out.X,2))).^2;
    norm_multi_z_z_new_2 = sum(((Out.X-X_new*ones(1,size(Out.X,2))).^2),1);
    norm_multi_z_z_new = sqrt(sum(((Out.X-X_new*ones(1,size(Out.X,2))).^2),1));
    
    %D_te_hat    = ((1-alpha)*D_te - (-1)*alpha*DY_beta_mod.*D_te);
    %FSum = sum(((D_te_hat - sqrt(sum(((Out.X-X_new*ones(1,size(Out.X,2))).^2),1))).^2).*H2,2);
    FSum = sum(((D_te - sqrt(sum(((Out.X-X_new*ones(1,size(Out.X,2))).^2),1))).^2).*H2,2);
    FSqSum = sum(-2*DY_mod.* D_te.*H2.*sqrt(sum(((Out.X-X_new*ones(1,size(Out.X,2))).^2),1)),2);
    
    F_crit = (1-alpha)*FSum + alpha*FSqSum;
    %F_crit = FSum;
    %F_crit = FSqSum;
end

function H2=H2_Gaussian(D,q)
H2=zeros( size(D,1), size(D,2));
for i = 1:size(D,1)
    for j = 1:size(D,2)
        H2(i,j)= exp(D(i,j)^2/(q*max(max(D(i,:)),max(D(:,j)))^2));
    end
end
end


function fn= FNorm(A)
% Compute the Frobenius norm of A, i.e., ||A||_F^2
    fn=sum(sum(A.*A));
end

function ip= InnerPro(A,B)
% Compute the inner product of A, i.e., ||A||_F^2
    ip=sum(sum(A.*B));
end