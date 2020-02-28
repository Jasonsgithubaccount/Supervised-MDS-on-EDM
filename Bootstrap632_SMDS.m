function [sol, model, Out_te_boot]= Bootstrap632_SMDS(  B, data, target, dim, alpha, q, disttype,  varargin )
% - estimate SMDS error rate using bootstrap .632+ rule
%
% USAGE: sol = f_cda632(data,target,method,B,verb)
%
% data      = matrix of training data (rows = obs, cols = variables) 
% target    = vector of integers specifying group membership of data
%
% B  = # iterations for bootstrap resampling   (default = 50)
% verb  = verbose display of results              (default = 1);
% 
% sol   = structure of results with the following fields
%  .err         = apparent error
%  .err_1       = bootstrap error
%  .err_632     = .632 bootstrap error
%  .err_632plus = .632+ bootstrap error
%  .R           = relative overfitting rate;
%  .gamma       = no-information rate;
%  .p           = prior probabilities;
%  .q           = posterior probabilities;
%
% SEE ALSO: f_cdaCV, f_cdaBCV, f_cda, classify

% -----References:-----
% Efron, B. and R. Tibshirani. 1997. Improvements on cross-validation: the
%   .632+ bootstrap method.
% Furlanello, C., S. Merler, C. Chemini, and A. Rizzoli. An application of
%   the bootstrap 632+ rule to ecological data.
% Ambroise, C. and G. J. McLachlan. 2002. Selection bias in gene extraction 
%   on the basis of microarray gene-expression data. Proc. Natl. Acad. Sci.
%   USA 99(10): 6562-6566.
%
% Equations refer to Efron & Tibshirani (1997) unless indicated otherwise.

% -----Author:-----
% by David L. Jones, Feb-2004
%
% This file is part of the FATHOM Toolbox for Matlab and is released under
% the GNU General Public License, version 2.

% -----Set defaults & check input:-----

if B < 0
    B   = 50; 
end  % default 50 bootstraps



if size(data,1) ~= size(target,1)
   error('data & target must have same # of rows!');
end
% ---------------------------------------

n = size(data,1); % # of observations
% data = f_stnd(data); % stardardize columnwise

% Apparent Error (training set == test set):
% switch method
%     case {1,2,3}
%         cls = classify(data,data,target,type,'empirical');
%         err = [sum(logical([cls - target] ~= 0))]/n;
%     case 4
%         cls = ;
% end


% Apparent Error (training set == test set):
data_train = data;
target_train = target;
data_test = data;
target_test = target;
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
Out_te_ae=SQREDM_SMDS_test_1vsall(model,cutpoint);
target_ae_hat = Out_te_ae.PY_te_hat_fi;
err = (sum(logical((target_ae_hat - target) ~= 0)))/n;
% -----No-information rate:-----
utarget  = unique(target);  % unique groups
notarget = size(utarget,1); % # groups

% Preallocate:
pi     = zeros(1,notarget);
qi     = pi;
gamma = 0;

for i=1:notarget % after Ambroise & McLachlan, 2002 (eq.3):
   pi_idx      = find(target==utarget(i));    % input
   qi_idx      = find(target_ae_hat==utarget(i));    % output
   pi_targetSize  = size(pi_idx,1);
   qi_targetSize  = size(qi_idx,1);
   pi(i)       = pi_targetSize/n;           % priors
   qi(i)       = qi_targetSize/n;           % posteriors
   gamma      = gamma + pi(i)*(1-qi(i)); % no-information rate
end
% ------------------------------


fid = fopen('Output.txt','a+');
fprintf(fid,'--------------------------------------------------------------\n\n');
fprintf(fid,'                  Bootstrap .632+ of SCMD                    \n\n');
fprintf(fid,'--------------------------------------------------------------\n');
fprintf(fid,'--------------------------------------------------------------\n');
fprintf(fid,'Iteration: B=%d \n',B);
fprintf(fid,'--------------------------------------------------------------\n\n');
fclose(fid);   

% -----Bootstrap:-----
err_boot = zeros(B,1); % preallocate
for b = 1:B      
   
   % Bootstrap training set:
   [idxTrain,idxTest] = f_boot(1:n);
      
   % Classify test set (Training and Test sets don't overlap):
   fid = fopen('Output.txt','a+');
   fprintf(fid,'\n\n--------------------------------------------------------------\n');
   fprintf(fid,'                     %dth iteration                           \n',b);
   fprintf(fid,'--------------------------------------------------------------\n\n');
   fclose(fid);
   
   
    if size(data,1) ~= size(data,2)     % when columns of 'data' represent features
        data_train      = data(idxTrain,:);
        target_train    = target(idxTrain,:);
        data_test       = data(idxTest,:);
        target_test     = target(idxTest,:);
    else                                % when 'data' is distance matrix
        data_train      = data(idxTrain,idxTrain);
        target_train    = target(idxTrain,:);
        data_test       = data(idxTest,idxTrain);
        target_test     = target(idxTest,:);
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
   Out_te_boot=SQREDM_SMDS_test_1vsall(model,cutpoint);
   target_boot_hat = Out_te_boot.PY_te_hat_fi;
   % Mis-classification Error (training ~= test set);
   err_boot(b) = sum(logical((target_boot_hat - target_test) ~= 0))/length(idxTest); 
end   

% Bootstrap error:
err_1 = mean(err_boot(:));
% --------------------   

% Bootstrap .632 (eq. 24):
err_632 = (0.368*err) + (0.632*err_1);

% Relative overfitting rate (eq. 28):
R = (err_1 - err)/(gamma - err);

% -----Make sure R ranges from 0-1:-----
err_1 = min([err_1 gamma]);

if (err_1 > err) && (gamma > err) % (eq. 31)
   % R = R;
else
   R = 0;
end
% --------------------------------------

% Bootstrap .632+ (eq. 32):
err_632plus = err_632 + (err_1 - err)*((0.368*0.632*R)/(1-0.368*R));

% Wrap results up into a structure:
sol.err         = err;
sol.err_1       = err_1;
sol.err_632     = err_632;
sol.err_632plus = err_632plus;
sol.R           = R;
sol.gamma       = gamma;
sol.pi           = pi;
sol.qi           = qi;
end



function [B,T] = f_boot(data,c,n)
% - bootstrap resampling with replacement
%
% USAGE: [B,T] = f_boot(data,c,n);
%
% data = input matrix                    (rows = obs, cols = variables)
% c = resample each column separately (default = 0);
% n = number of bootstrapped samples  (default = same as input data)
%
% B = bootstrapped sample
% T = elements of data not in B
%
% SEE ALSO: f_targetBoot, f_shuffle, f_randRange

% -----Author:-----
% by David L. Jones, Dec-2003
%
% This file is part of the FATHOM Toolbox for Matlab and
% is released under the GNU General Public License, version 2.

% Feb-2004: now uses unidrnd vs. f_randRange, added output of T
% Jan-2010: replaced '&' with '&&'
% Oct-2010: now works with matrices, can specify n, optional resample columns
%           separately

% -----Set defaults and check input:-----
if (size(data,1)==1), data = data(:);    end % handle row vectors
if (nargin < 2), c = 0;         end % default don't resample cols separately
if (nargin < 3), n = size(data,1); end % default same # of samples as data
% -----------------------------------------

[nr,nc] = size(data);
B       = repmat(NaN,n,nc); % preallocate

if (c>0) % Resample each column separtely:
   for i = 1:nc
      B(:,i) = data(unidrnd(nr,n,1),i);
   end
   T = NaN; % not applicable here
else % Resample obs with replacement:
   B = data(unidrnd(nr,n,1),:);
   T = setdiff(data,B,'rows'); % elements not resampled
end
end