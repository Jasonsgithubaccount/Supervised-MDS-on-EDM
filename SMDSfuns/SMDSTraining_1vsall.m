function [model,cutpoint]=SMDSTraining_1vsall(data_train, target_train,...
    data_test, target_test, dim, alpha, q, distype, varargin)
%SMDSTraining_1vsall  Supervised multi-dimensional scale training based on
%1 vs all classification.
% 
%INPUTS:
%   data_train:        
%   varargin:      When string 'distance' is in the first cell, it means 
%                   that the data_train is distance matrix; if otherwise,
%                   data_train is original feature data. If there is extra
%                   parameter requirement, the first or the second cell of
%                   'varargin' will contain the structure of 'par_extra'
%                   which contains the relevant extra parameters.
%                 
%   input:        n*d matrix,representing samples
%   target:       n*1 matrix,class label
%   model:        struct type(see codes below)
%   k:            the total class number          
%   ClassLabels:   the class name of each class
%
%% 


fid = fopen('Output.txt','a+');
while fid ==-1
            fid=fopen('Output.txt','a+');
end
fprintf(fid,'--------------------------------------------------------------\n\n');
fprintf(fid,'                        SCMD   Training                       \n\n');
fprintf(fid,'--------------------------------------------------------------\n');
fprintf(fid,'Parameters of SCMD : dim   = %d;\n', dim);
fprintf(fid,'                     alpha = %1.3f;\n', alpha);
fprintf(fid,'                     q     = %1.3f.\n',  q);
fprintf(fid,'--------------------------------------------------------------\n\n');
fclose(fid);
model=struct;
ClassLabels=unique(target_train);
T=length(ClassLabels);
if  ~ismember('distance',varargin)
    if ismember('extrapar',varargin)
        par_extra = cell2struct(varargin(find(ismember(varargin,'extrapar'),1)+1));
        [D, pars, D_te] = EDM_PARConstruction(data_train, target_train, data_test, target_test, distype, par_extra);
    else 
        [D, pars, D_te] = EDM_PARConstruction(data_train, target_train, data_test, target_test, distype);
    end
else
    if size(data_train,1) ~= size(data_train,2) 
        error('Incorect distance matrix!')
    end
    D       = data_train;
    D_te    = data_test;
    if ismember('extrapar',varargin)
        par_extra = cell2struct(varargin(find(ismember(varargin,'extrapar'),1)+1));
        if isfield(par_extra, 'range')
            pars.range = par_extra.range;
        end
        if isfield(par_extra, 'LOWBD') 
            pars.LOWBD = par_extra.LOWBD;
        end
        if isfield(par_extra, 'UPPBD') 
            pars.UPPBD = par_extra.UPPBD;
        end
    end
    pars.PP      = [];
    pars.PY      = target_train';
    pars.PS_te   = [];
    pars.PY_te   = target_test';
end
lable_b = max(pars.PY)+1;
for t=1:T
        model(t).a=t;
        lable_a = ClassLabels(t);
        model(t).lable_a = lable_a;
        fid = fopen('Output.txt','a+');
        while fid ==-1
            fid=fopen('Output.txt','a+');
        end
        fprintf(fid,'-------------------Model Training-----------------------------\n');
        fprintf(fid,'                   Class_a Label:%1d          \n', ClassLabels(t));
        fprintf(fid,'--------------------------------------------------------------\n');
        fclose(fid);
        g =(pars.PY==lable_a);
        D_2c=D;
        n_2c = size(D_2c,1);
        target_train_2c= pars.PY';
        target_train_2c(~g) = lable_b;        
        model(t).lable_b = lable_b;
        DY_2c      = zeros(n_2c,n_2c);
        DY_2c(target_train_2c*ones(1,n_2c) ~= (target_train_2c*ones(1,n_2c))')=1;
        D_te_2c=D_te;
        
        pars_2c = pars;
        pars_2c.PY      = target_train_2c'; 
        Out = SQREDM_SMDS(D_2c,dim,pars_2c,DY_2c,alpha,q);
        model(t).Out = Out;
        model(t).D_te= D_te_2c;
        model(t).pars_2c= pars_2c;
end
cutpoint=SQREDM_SMDS_test_cutpoint(model,D);
end

function D_ba = matrixbalancing()
% imbalance data reconstruction
        %n_2c = size(D_2c,1);
%         n_a = numel(find(pars_2c.PY==lable_a));
%         n_b = numel(find(pars_2c.PY==lable_b));
%         D_l=[];
%         if n_a > n_b
%               g_t   = find(pars_2c.PY == lable_b);
%           for l = 1:n_2c
%              D_l(l) = sqrt(((n_b+1) * sum(D_2c(l,g_t).^2)-0.5*sum(sum(D_2c(g_t,g_t).^2)))/(n_b^2));
%           end
%           DY_l = ones(1,n_2c);
%           DY_l(g_t) = 0;
%           for l = 1: n_a - n_b
%               D_2c(l+n_2c,:) = D_l;
%               pars_2c.PY(l+n_2c) = lable_b;
%               DY_2c(l+n_2c,:) = DY_l;
%           end
%           for l = 1: n_a - n_b
%               D_2c(:,l+n_2c) = [D_l,zeros(1,n_a - n_b)]';
%               DY_2c(:,l+n_2c) = [DY_l,zeros(1,n_a - n_b)]'; 
%           end
%           D_te_2c = [D_te_2c,zeros(n_te,n_a - n_b)];
%           for l = 1 : n_te
%               D_te_2c(l,n_2c+1: n_2c + n_a - n_b) = sqrt(((n_b+1) * sum(D_te_2c(l,g_t).^2)-0.5*sum(sum(D_2c(g_t,g_t).^2)))/(n_b^2));
%           end
%           
%         end
%         if n_a < n_b
%               g_t   = find(pars_2c.PY==lable_a);
%           for l = 1:n_2c
%              D_l(l) = sqrt(((n_a+1) * sum(D_2c(l,g_t).^2)-0.5*sum(sum(D_2c(g_t,g_t).^2)))/(n_a^2));
%           end
%           DY_l = ones(1,n_2c);
%           DY_l(g_t) = 0;
%           for l = 1: n_b - n_a
%               D_2c(l+n_2c,:) = D_l;
%               pars_2c.PY(l+n_2c) = lable_a;
%               DY_2c(l+n_2c,:) = DY_l;
%           end
%           for l = 1: n_b - n_a
%               D_2c(:,l+n_2c) = [D_l,zeros(1,n_b - n_a)]';
%               DY_2c(:,l+n_2c) = [DY_l,zeros(1,n_b - n_a)]'; 
%           end
%           D_te_2c = [D_te_2c,zeros(n_te,n_b - n_a)];
%           for l = 1 : n_te
%               D_te_2c(l,n_2c+1: n_2c + n_b - n_a) = sqrt(((n_a+1) * sum(D_te_2c(l,g_t).^2)-0.5*sum(sum(D_2c(g_t,g_t).^2)))/(n_a^2));
%           end
%         end
%         
%              f_a = find(pars_2c.PY==lable_a);
%              f_b = find(pars_2c.PY==lable_b);
%              n_a = numel(f_a);
%              n_b = numel(f_b);
%               
%                  [~,cidx_a] = kmedioids(D_2c(f_a,f_a),n_c);
%                  [~,cidx_b] = kmedioids(D_2c(f_b,f_b),n_c);
%                  f_t=[f_a(cidx_a),f_b(cidx_b)];
%                  D_2c = D_2c(f_t,f_t);
%                  pars_2c.PY = pars_2c.PY(f_t);
%                  DY_2c = DY_2c(f_t,f_t);
%                  D_te_2c = D_te_2c(:,f_t);
             

%              if n_b > n_c        
%                  [~,cidx] = kmedioids(D_2c(f_b,f_b),n_c);
%                  f_t=[f_a,f_b(cidx)];
%                  D_2c = D_2c(f_t,f_t);
%                  pars_2c.PY = pars_2c.PY(f_t);
%                  DY_2c = DY_2c(f_t,f_t);
%                  D_te_2c = D_te_2c(:,f_t);
%              end        
end
