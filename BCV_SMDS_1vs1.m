function [err, model, Out_te_loo]= BCV_SMDS_1vs1(  B, data, target, dim, alpha, q, disttype,  varargin )
%% bootstrap cross-validation for SMDS
%%
% USAGE: f_cdaBCV(data,target,method,B)
%
% data   = input data (rows = observations, cols = variables)
% target = column vector of integers specifying group membership
%
%
% B  = # iterations for bootstrap resampling (default = 50)
%%
% SEE ALSO: LOO_SMDS, f_cda632, classify, f_grpBoot

% -----Notes:-----
% This function is used to diagnose a Canonical Discriminant Analysis using
% a new bootstrap method of cross-validation, specifically for small sample
% sizes.

% -----References:-----
% Fu, W. J., R. J. Carroll, and S. Wang. 2005. Estimating misclassification
%   error with small samples via bootstrap cross-validation. Bioinformatics
%   21(9): 1979-1986.

% -----Author:-----
% by David L. Jones, Oct-2010
%
% This file is part of the FATHOM Toolbox for Matlab and is released under
% the GNU General Public License, version 2.

%% Set defaults & check input:-----
if (nargin < 4), B   = 50; end % default 50 bootstraps


target = target(:);    % force col vector
n   = size(data,1); % # of rows (observations)

if (n ~= numel(target))
   error('# of rows in data and target must be equal !');
end

if (B<50), error('B should be from 50-200!'); end
% -------------------------------------

utarget   = f_unique(target);  % unique groups, unsorted
notarget  = size(utarget,1);   % # of groups

% Preallocate:
err.tot      = repmat(NaN,B,1);
err.grp      = repmat(NaN,notarget,B);

% Repeat for each bootstrap sample:

fid = fopen('Output.txt','a+');
fprintf(fid,'--------------------------------------------------------------\n\n');
fprintf(fid,'             Bootstrap Cross-validation of SCMD               \n\n');
fprintf(fid,'--------------------------------------------------------------\n');
fprintf(fid,'--------------------------------------------------------------\n');
fprintf(fid,'Iteration: B=%d \n',B);
fprintf(fid,'--------------------------------------------------------------\n\n');
fclose(fid);
for j = 1:B
   
   % Create boostrap sample:
   data_train = f_grpBoot(data,target,0);
   target_train = target;
   % Leave-One-Out Cross-Validation of Bootstrapped data:
   fid = fopen('Output.txt','a+');
   fprintf(fid,'\n\n--------------------------------------------------------------\n');
   fprintf(fid,'                 BCV    %dth interation                         \n',j);
   fprintf(fid,'--------------------------------------------------------------\n\n');
   fclose(fid);
   
    if  ~ismember('distance',varargin)
        if ismember('extrapar',varargin)
         par_extra = cell2struct(varargin(find(ismember(varargin,'extrapar'),1)+1));
         [errB, model, Out_te_loo] = LOO_SMDS_1vs1(data_train, target_train, ...
              dim, alpha,q, disttype,'extrapar', par_extra);
        else 
         [errB, model, Out_te_loo] = LOO_SMDS_1vs1(data_train, target_train, ...
              dim, alpha, q, disttype);
        end
    else
        if ismember('extrapar',varargin)
         par_extra = cell2struct(varargin(find(ismember(varargin,'extrapar'),1)+1));
         [errB, model, Out_te_loo] = LOO_SMDS_1vs1(data_train, target_train, ...
              dim, alpha,q, disttype,'distance','extrapar', par_extra);
        else 
         [errB, model, Out_te_loo] = LOO_SMDS_1vs1(data_train, target_train, ...
              dim, alpha, q, disttype,'distance');
        end
    end
   
   
   
   
   
      
   % Collect classification error rates for each iteration:
   err.tot(j)         = errB.tot;
   err.grp(1:notarget,j) = errB.grp;
end

% Average results:
err.tot = mean(err.tot);
err.grp = mean(err.grp,2);


end


function B = f_grpBoot(X,grp,c)
% - within-group bootstrap sampling (fixed size)
%
% USAGE: B = f_grpBoot(X,grp,c)
%
%  X   = input data (rows = observations, cols = variables)
% grp  = column vector of integers specifying group membership
% c    = resample each column separately (default = 0);
%
% B    = boostrapped version of X
%
% SEE ALSO: f_grpResample, f_boot, f_shuffle

% -----Notes:-----
% This function performs bootstrapped resampling with replacement. The
% resampling is performed separately for each group and can also optionally be
% performed separately for each column (variable) within each group. This last
% option reduces the level of duplicate observations created in the bootstrapped
% sample.

% -----References:-----
% Fu, W. J., R. J. Carroll, and S. Wang. 2005. Estimating misclassification
%   error with small samples via bootstrap cross-validation. Bioinformatics
%   21(9): 1979-1986.

% -----Author:-----
% by David L. Jones, Oct-2010
%
% This file is part of the FATHOM Toolbox for Matlab and
% is released under the GNU General Public License, version 2.

% -----Set defaults & check input:-----
if (nargin < 3), c = 0; end % default bootstrap columns separately

grp = grp(:);    % force col vector
n   = size(X,1); % # of rows (observations)

if (n ~= numel(grp))
   error('# of rows in X and GRP must be equal !');
end
% -------------------------------------

uGrp   = f_unique(grp);  % unique groups, unsorted
noGrp  = size(uGrp,1);   % # of groups

% Preallocate:
idx.G{noGrp} = NaN;
idx.n{noGrp} = NaN;
B            = repmat(NaN,size(X));

% Get indices to members of each group:
for i=1:noGrp
   idx.G{i} = find(grp == uGrp(i));
   idx.n{i} = numel(idx.G{i});
   if idx.n{i} < 3
      error('Each group must have at least 3 members!')
   end
end

% Create boostrap sample:
for i = 1:noGrp
   if (c>0) % Resample each column separately:
      B(idx.G{i},:) = f_boot(X(idx.G{i},:),1);
      
   else % Resample observations:
      % Initialize each group with 3 distinct obs (Fu et al., 2005):
      boot               = f_shuffle(X(idx.G{i},:),4);  % shuffle obs of this group
      B(idx.G{i}(1:3),:) = boot(1:3,:);                 % initialize with 3 of these
      
      % Fill in remaining obs:
      if idx.n{i}>3
         B(idx.G{i}(4:end),:) = f_boot(boot,0,idx.n{i}-3); % boostrap obs
      end
   end
end

end


function y = f_shuffle(x,method,grp)
% - randomly sorts vector, matrix, or symmetric distance matrix
%
% Usage: y = f_shuffle(x,method,grp) 
% 
% -----Input/Output:-----
% x      = vector, matrix, or symmetric distance matrix
%
% method = type of permutation to perform
%          (default = 1 for a regular matrix or vector)
%          (default = 2 for a symmetric distance matrix)
%          1: unrestricted permutation
%          2: unrestricted, rows & cols are permuted the same way
%          3: permutation restricted to within columns of matrix
%          4: permute order of rows only (works across the matrix)
%          5: permutation restricted to within groups defined by grp
%
% grp    = optional vector of integers specifying group membership for
%          restricted permutation 
%
% y      = random permutation of x
%     
% SEE ALSO: f_boot, f_grpBoot, f_randRange

% -----Notes:-----
% When permuting a symmetric distance matrix, care must be taken to shuffle the
% objects making up the matrix and not the tridiagonal (see references below)
% 
% To initialize RAND to a different state, call the following at the beginning
% of your routine (but not within a loop):
% >> rand('twister',sum(100*clock));


% -----References (for permutation of distance matrix):-----
% Legendre, P. & L. Legendre. 1998. Numerical ecology. 2nd English ed.
%  Elsevier Science BV, Amsterdam. xv + 853 pp. [page 552]
% Sokal, R. R. and F. J. Rohlf. 1995. Biometry - The principles and 
%  practice of statistics in bioligical research. 3rd ed. W. H. 
%  Freeman, New York. xix + 887 pp. [page 817]

% ----- Author: -----
% by David L. Jones, Mar-2002
%
% This file is part of the FATHOM Toolbox for Matlab and
% is released under the GNU General Public License, version 2.

% 31-Mar-2002: added restricted permutation via grouping vector
% 18-Apr-2002: added switch-case handling of method options
%              and column-restricted permutation
% Dec-2007:    edited documentation
% Jan-2008:    changed & to &&; initialize RAND each time, changed 'find' to
%              logical indexing in 'case 5', preallocation of Y.
% Apr-2008:    Don't initialize RAND each time.
% Oct-2010:    Method 3 is now done internally

% -----Set Defaults:-----
if (nargin<2) && (f_issymdis(x)==0), method = 1; end; % default for non-symmetric matrices
if (nargin<2) && (f_issymdis(x)==1), method = 2; end; % default for symmetric matrices
% -----------------------

[nr,nc] = size(x);
y = zeros(nr,nc); % preallocate

switch method   
case 1 % Permutation of a regular vector or matrix:
   y = x(randperm(length(x(:))));
   y = reshape(y,nr,nc);
   
case 2 % Permutation of rows then colums,in the same way
   if (nr~=nc)
      error('Method 2 requires a square matrix')
   end
   i = randperm(nr); % get permuted indices
   y = x(i,:);       % permute rows
   y = y(:,i);       % permute cols
   
   case 3 % Permutation restricted to columns
      %    for i = 1:nc
      %       y(:,i) = f_shuffle(x(:,i),1);
      %    end
      for i = 1:nc
         y(:,i) = x(randperm(nr),i);
      end
   
case 4 % Permute order of rows only (works across the matrix)
   i = randperm(nr); % get permuted indices
   y = x(i,:);       % permute rows
   
case 5 % Permutation restricted to groups:
   if (nargin<3)
      error('Restricted permutation requires a grouping vector');
   end
   
   % make sure inputs are compatible:
   if (prod((size(x) == size(grp)))==0);
      error('X & GRP are not of compatible sizes');
   end;
   
   uGrp  = unique(grp);    % unique groups
   noGrp = length(uGrp); % # of groups
   
   for i = 1:noGrp
      % y(find(grp==uGrp(i))) = x(f_shuffle(find(grp==uGrp(i))),1);
      y(grp==uGrp(i)) = x(f_shuffle(grp==uGrp(i)),1);
   end;
   
   % Return to column vector if necessary:
   if (size(x,1)>1) && (size(x,2)==1), y = y(:); end; 
   
otherwise  
   error('Unknown permutation method!');
end


end


function [err, model, Out_te_loo] = LOO_SMDS_1vs1(  data, target, dim, alpha, q, disttype,  varargin )
% - leave-one-out cross validation for Canonical Discriminant Analysis
%
% USAGE: [err,PP] = LOO_SMDS(data,grp,method,verb);
%
% data   = input data (rows = objects, cols = variables)
% grp = column vector of integers specifying group membership
%

% 
% err = structure of results having the following fields:
%  .tot  = total error
%  .grp  = total error for each group
%  .conf = confusion matrix (proportion of CLS classified as GRP)
% 
% PP = posterior probabilities
%
% SEE ALSO: f_cda, f_cda632, classify

% -----Notes:-----
% This function is used to diagnose a Canonical Discriminant Analysis using
% the Leave-One-Out method of Cross-Validation. For methods 1-3, most of
% the work is done by the CLASSIFY function in the Matlab Statistics
% Toolbox. Method 4 ('centroid') follows Anderson (2002) and Anderson &
% Willis (2003).

% -----References:-----
% Anderson, M. J. 2002. CAP: a FORTRAN program for canonical analysis of
%  principal coordinates. Dept. of Statistics University of Auckland.
%  Available from: http://www.stat.auckland.ac.nz/PEOPLE/marti/
% Anderson, M. J. & T. J. Willis. 2003. Canonical analysis of principal
%  coordinates: a useful method of constrained ordination for ecology.
%  Ecology 84(2): 511-525.
% White, J. W. and B. I. Ruttenberg. 2007. Discriminant function analysis in
%  marine ecology: some oversights and their solutions. Mar. Ecol. Prog. Ser.
%  329: 301-305.

% -----Author:-----
% by David L. Jones, Dec-2003
%
% This file is part of the FATHOM Toolbox for Matlab and is released under
% the GNU General Public License, version 2.

% modified after f_capValid, written April-2003
% Feb-2004: re-writted to call f_errRate, now calculates confusion matrix,
%           fixed setting of defaults
% Oct-2010: added support for method = 4 ('centroid'), adapted from code
%           originally in f_capValid and f_cdaValid; changed default method to
%           4.
% Feb-2011: replaced 'unique' with 'f_unique'; use non-informative priors
%           following recommendations of Wright & Ruttenberg; return
%           posterior probabilities; added support for spatial median


% -----------------------

n      = size(data,1);               % # of rows (observations)

target_hat    = repmat(NaN,n,1);         % preallocate


for omit = 1:n
   idx       = 1:n; % index of all observations
   idx(omit) = [];  % leave one observation out
   fid = fopen('Output.txt','a+');
   fprintf(fid,'\n\n--------------------------------------------------------------\n');
   fprintf(fid,'                 LOO    %dth observation                         \n',omit);
   fprintf(fid,'--------------------------------------------------------------\n\n');
   fclose(fid);
   if size(data,1) ~= size(data,2)     % when columns of 'data' represent features
        data_train      = data(idx,:);
        target_train    = target(idx,:);
        data_test       = data(omit,:);
        target_test     = target(omit,:);
   else                                % when 'data' is distance matrix
        data_train      = data(idx,idx);
        target_train    = target(idx,:);
        data_test       = data(omit,idx);
        target_test     = target(omit,:);
   end
   if  ~ismember('distance',varargin)
        if ismember('extrapar',varargin)
         par_extra = cell2struct(varargin(find(ismember(varargin,'extrapar'),1)+1));
         [model,cutpoint] = SMDSTraining_1vs1(data_train, target_train, ...
             data_test ,target_test, dim, alpha,q, disttype,'extrapar', par_extra);
        else 
         [model,cutpoint] = SMDSTraining_1vs1(data_train, target_train, ...
             data_test ,target_test, dim, alpha, q, disttype);
        end
   else
        if ismember('extrapar',varargin)
         par_extra = cell2struct(varargin(find(ismember(varargin,'extrapar'),1)+1));
         [model,cutpoint] = SMDSTraining_1vs1(data_train, target_train, ...
             data_test ,target_test, dim, alpha,q, disttype,'distance','extrapar', par_extra);
        else 
         [model,cutpoint] = SMDSTraining_1vs1(data_train, target_train, ...
             data_test ,target_test, dim, alpha, q, disttype,'distance');
        end
   end
   Out_te_loo=SQREDM_SMDS_test_1vs1(model,cutpoint); 
   target_loo_hat = Out_te_loo.PY_te_hat_fi;
   target_hat(omit) = target_loo_hat;
%    if (~isequal(type,'centroid')) && (~isequal(type,'spatial median'))
%       [cls_loo,null,PP_loo] = classify(x(omit,:),x(idx,:),grp(idx),type,priors);
%       cls(omit)  = cls_loo;
%       PP(omit,:) = PP_loo;
%    
%    else
%       % Use Centroid or Spatial Median:
%       if (method==4), sm = 0; elseif (method==5), sm = 1; end
%       
%       % Perform CDA with 1 obs omitted, unwrap structure of results:
%       result_loo = f_cda(x(idx,:),grp(idx),1,0,0,sm);
%       Cvects     = result_loo.Cvects;
%       centroids  = result_loo.centroids;
%       
%       % Center omitted obs using means of others:
%       loo_ctr = x(omit,:) - mean(x(idx,:));
%       
%       % Project omitted obs in canonical space:
%       loo_scores = loo_ctr*Cvects;
%       
%       % Find group centroid closest to omitted obs and classify accordingly:
%       D = sum((repmat(loo_scores,noGrp,1) - centroids).^2,2)'; % squared distances
%       L                = 1 - (D / max(D(:)));  % convert distance to likelihood
%       PP(omit,:)       = L / sum(L);           % posterior probabilities
%       [null,cls(omit)] = max(PP(omit,:),[],2); % classify omitted obs
%    end
end

% Classification error rates:
err = f_errRate(target,target_hat);

% % -----Send output to display:-----
% if (verb>0)
%    fprintf('\n==================================================\n');
%    fprintf('            F_CDA CROSS-VALIDATION\n'                 );
%    fprintf('            Classification Success: \n'               );
%    fprintf('--------------------------------------------------\n' );
%    
%    fprintf('Group        Corrrect  \n');
%    for j=1:noGrp
%       fprintf('%s %d %s %10.1f %s \n',['  '],uGrp(j),['     '],[1-err.grp(j)]*100,['%']);
%    end
%    
%    fprintf('\n\n');
%    fprintf('Total Correct  = %4.2f %s \n',(1-err.tot)*100,['%']);
%    fprintf('Total Error    = % 4.2f %s \n',err.tot*100,['%']);
%    fprintf('Class. method  = %s \n',type);
%    if method<4, fprintf('Priors = non-informative\n'); end
%    fprintf('\n--------------------------------------------------\n' );
%    fprintf('     Confusion Matrix (%s): \n',['%'])
%    hdr = sprintf('%-6.0f ',uGrp(:)');
%    fprintf(['group: ' hdr]);
%    fprintf('\n')
%    for j=1:noGrp
%       txt = [sprintf('%6.0f ',uGrp(j)) sprintf('%6.1f ',err.conf(j,:)*100)];
%       fprintf(txt);
%       fprintf('\n')
%    end
%    fprintf('\n==================================================\n\n');
% end
end


function err = f_errRate(grp,cls)
% - error rate for a classifier
%
% USAGE: err = f_errRate(grp,cls)
%
% grp = input vector of integers specifying group membership
% cls = group membership predicted by the classifier
%
% err = structure of results having the following fields:
% err.tot  = total error
% err.grp  = total error for each group
% err.uGrp = list of groups
% err.conf = confusion matrix (proportion of CLS classified as GRP)
%
% SEE ALSO: LOO_SMDS, f_boot632, f_chanceClass

% -----Notes:-----
% This function is used to determine error rates for classifiers; in
% particular, it is called by LOO_SMDS. More robust measures of
% generalization error may be achieved using f_boot632, etc.

% -----Author:-----
% by David L. Jones, Feb-2004
%
% This file is part of the FATHOM Toolbox for Matlab and
% is released under the GNU General Public License, version 2.

% Feb-2011: replaced 'unique' with 'f_unique'; added uGrp to output

% -----Check input:-----
grp = grp(:);
cls = cls(:);

if size(grp,1) ~= size(cls,1)
   error('GRP and CLS must be same size!')
end
% -----------------------

n     = size(grp,1);   % # obs
uGrp  = f_unique(grp); % unique groups, unsorted
noGrp = size(uGrp,1);  % # of groups

% Preallocate:
err.tot  = NaN;
err.grp  = zeros(noGrp,1);
err.uGpr = uGrp;
err.conf = zeros(noGrp,noGrp);

for i=1:noGrp
   idx     = find(grp==uGrp(i));
   grpSize = size(idx,1);
   
   % Confusion matrix:
   for j=1:noGrp
      err.conf(i,j) = [sum(logical(cls(idx) == uGrp(j)))]/grpSize;
   end   
end

% Total error rate:
err.tot = [sum(logical([grp - cls] ~= 0))]/n; % 1=pass, 0=fail

% Error rate by group:
err.grp = diag([1 - err.conf]);
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


function U = f_unique(X)
% - returns unsorted list of unique values
% 
% USAGE: U = f_unique(X)
% 
% X = iput matrix
% U = unsorted list of unique values in X
% 
% SEE ALSO: unique

% -----References:-----
% http://www.mathworks.de/matlabcentral/newsreader/view_thread/236866

% -----Author:-----
% by David L. Jones, Sep-2010
%
% This file is part of the FATHOM Toolbox for Matlab and
% is released under the GNU General Public License, version 2.

% Jan-2011: added support for cell arrays

if iscell(X)
   [nul,I] = unique(X,'first');
else
   [nul,I] = unique(X,'rows','first');
end

U = X(sort(I),:);
end
