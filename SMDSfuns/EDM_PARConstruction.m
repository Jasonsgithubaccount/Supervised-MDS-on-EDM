function [D, pars, D_te] = EDM_PARConstruction(data_train, target_train,...
    data_test, target_test, disttype, par_extra)
%EDM_PARConstruction
% 
% INPUTS:
%
% par_extra: parameters for extra demands of distance matrix
%       noisetype = 'additive', 'multiplicative', 'log-normal'
%       n: number of anchors and sensors
%       m: number of anchors
%       nf: 0.1
%       range: used to construct the initial (perturbed) distances
%
% OUTPUTS:
%
% D = [anchor-anchor (squared) distance, anchor-sensor (squared) distance;
%      sensor-anchor (squared) distance, sensor-sensor (squared) distance]
%      distances are squared    
%      diag(D) = 0
%
% DY = 

if nargin == 6
    if isfield(par_extra, 'B') && isfield(par_extra, 'B_target')
        B = par_extra.B;
        B_target = par_extra.B_target;
    end
    if isfield(par_extra, 'noisetype') && isfield(par_extra, 'nf')
        noisetype = par_extra.noisetype;
        nf = par_extra.nf;
    end
    if isfield(par_extra, 'range')
        range = par_extra.range;
    end
    if isfield(par_extra, 'LOWBD') 
        LOWBD = par_extra.LOWBD;
    end
    if isfield(par_extra, 'UPPBD') 
        UPPBD = par_extra.UPPBD;
    end
    
else
    B = [];
    B_target = [];
    range = -1;
end

n       = size(data_train,1);
PB      = B';
m       = size(B,1);
t       = size(data_test,1);
PY      = [B_target',target_train'];
PY_te   = target_test';
PS      = data_train';
PS_te   = data_test';
PP      = [PB PS];

    
% generate the squared pre-distance matrix D
% use randidtance.m (Toh) to construct perturbed distances


if nargin == 6
    rng('default');
    randstate = rng('shuffle');
    D_all   = randistance_qi(PB,[PS,PS_te],range,nf,noisetype,randstate);
    D       = D_all(1:n,1:n);
    D_te    = D_all(n+1:n+t,1:n);
    [~,bins] = graphconncomp(sparse(D));
    if ~all(bins == 1)
    error('Graph is not connected, try to increase range...')
    end
else
    if strcmp(disttype, 'dtw_dist')
        data = [PS,PS_te]';
        n_to = size(data,1);
        D_all=zeros(n_to,n_to);
        for i = 1:n_to
            D_all(i,:)   = pdist2(data(i,:),data, @dtw_dist);
            if  mod(i,10)==0
                fprintf('Item %d distance caldulated\n',i);
            end
        end
        
    else
        if strcmp(disttype, 'minkowski')
            D_all   = squareform(pdist([PS,PS_te]', disttype, 1.5));
        else
            D_all   = squareform(pdist([PS,PS_te]', disttype));
        end
    end
    D       = D_all(1:n,1:n);
    D_te    = D_all(n+1:n+t,1:n);
end


if m > 0
    D0 = squareform(pdist(PB,disttype));
    D  = [D0 D_all(1:n, (n+t+1):n+t+m)'; D_all(1:n, (n+t+1):(n+t+m)) D_all(1:n, 1:n)];
end




pars.m       = m;
pars.range   = range;
pars.PP      = PP;
pars.PY      = PY;
pars.PS_te   = PS_te;
pars.PY_te   = PY_te;
%figure;scatter(PS(1,:),PS(2,:),[],PY(1,:)); hold on;

%if size(PS_te,1)~=0
%scatter(PS_te(1,:),PS_te(2,:),[],PY_te(1,:),'filled','h');
%end
%title(['Simulation Model'],...
%                    'FontName','Times','FontSize',8);
end



% This is the code from Toh's SNLSDP package
% The only difference is that the multiplicative perturbation 
% used abs 9used in Tseng's paper (Tseng SIOPM 2007).
%
%%*************************************************************************
%% Compute pair-wise distances within R limit between known-unknown and
%% unknown-unknown pairs.
%%
%% Dall = randistance(P0,PP,Radius,nf,noisetype);
%%
%% P0 : anchor positions
%%      nfix being the number of anchors
%% PP : sensor positions 
%%      npts being the number of unknown sensors
%% nf : noise factor 
%% noisetype : 1 - Normal :  dnoisy = dactual + N(0,1)*nf (Default)
%%             2 - Multiplicative Normal : dnoisy= dactual*(1 + N(0,1)*nf)
%%             3 - Log Normal : dnoisy= dactual * 10^(N(0,1)*nf)
%% Dall = [DD, D0]
%% DD = [sensor-sensor distance];
%% D0 = [anchor-sensor distance];
%%*************************************************************************

  function [Dall] = ...
            randistance_qi(P0,PP,Radius,nf,noisetype,randstate)

  if ~exist('noisetype'); noisetype = 'additive'; end
  if ~exist('randstate'); randstate = 0;
      rng('shuffle');
  end
  %if Radius <= 0; Radius=inf; end
%  randn('state',randstate);
%  rand('state',randstate);
%  rng(randstate);
  
  [~,npts] = size(PP);
  nfix = size(P0,2);
  D = pdist2([P0,PP]',[P0,PP]');
  D0 = D((nfix+1):(nfix+npts),1:nfix);
  DD = D((nfix+1):(nfix+npts),(nfix+1):(nfix+npts)); 
%%
  if strcmp(noisetype,'additive'); noisetype = 1; end
  if strcmp(noisetype,'multiplicative'); noisetype = 2; end
  if strcmp(noisetype,'log-normal'); noisetype = 3; end
%%
if Radius > 0;
    for j = 1:npts    
         if (nfix > 0)
            %tmp = PP(:,j)*ones(1,nfix)-P0;
            %rr = sqrt(sum(tmp.*tmp));      
            idx = find(D0(j,:) < Radius);
            %rr = D0(j,idx); 
            if (~isempty(idx))
                if (noisetype == 1)
                    D0(j,find(D0(j,:) < Radius)) = D0(j,find(D0(j,:) < Radius)) + (randn(1,length(idx))*nf);
                elseif (noisetype == 2)
                    D0(j,find(D0(j,:) < Radius)) = D0(j,find(D0(j,:) < Radius)) .*((1+(randn(1,length(idx))*nf))); % abs used
                elseif (noisetype == 3)
                    D0(j,find(D0(j,:) < Radius)) = D0(j,find(D0(j,:) < Radius)) .* 10.^(1+(randn(1,length(idx))*nf));
                end
                %D0(j,idx) = rr';
            end
         end        
         if (j > 1)
            %tmp = PP(:,j)*ones(1,j-1) - PP(:,1:j-1);
            %rr = sqrt(sum(tmp.*tmp));
            %rr = pdist2(PP',PP');
            idx = find(rr < Radius);
            %rr = rr(idx); 
            if (~isempty(idx))
                if (noisetype == 1)
                    DD(j,find(DD(j,1:(j-1)) < Radius)) = DD(j,find(DD(j,1:(j-1)) < Radius)) + (randn(1,length(idx))*nf);
                elseif (noisetype == 2)
                    DD(j,find(DD(j,1:(j-1)) < Radius)) = DD(j,find(DD(j,1:(j-1)) < Radius)) .*((1+(randn(1,length(idx))*nf))); % abs used
                elseif (noisetype == 3)
                    DD(j,find(DD(j,1:(j-1)) < Radius)) = DD(j,find(DD(j,1:(j-1)) < Radius)) .* 10.^(1+(randn(1,length(idx))*nf));
                end
                %DD(idx,j) = rr';
            end
         end
    end
end
  DD = triu(DD,1) + triu(DD,1)';
  Dall = [DD, D0];

end