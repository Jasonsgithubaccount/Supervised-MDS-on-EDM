function Out = SQREDM_SMDS(D,dim,pars,DY,alpha,q)
%
% This code aims to solve the model
%
%   min_Z  || sqrt(H2).*(sqrt(Z)-D) ||^2 + (rho/2) ||Z+P_Kr(-Z)||^2
%    s.t.    L<=Z<=U
%
%
% INPUTS:
%
%	D   : n-by-n dissimilarities matrix                          [required]
%         diag(D) = 0
%         dissimilarities are UNSQUARED, i.e.,
%                          D_ij=||point_i-point_j||+noise 
%         
%   dim : the embedding dimension  (e.g. dim = 2 or 3)           [required]
%
%   pars: parameters and other information                       [optional]
%         pars.m   : m  -- the number of given points, m>=0  
%         pars.PP  : PP -- dim-by-n matrix of coordinates of n points with
%                          first m(>=0) columns being given
%                    For sensor network localization (SNL)
%                    PP = [PA, PS]
%                    PA -- dim-by-m matrix of coordinates of m anchors
%                    PS -- dim-by-(n-m) matrix of coordinates of (n-m) sensors
%        pars.rho  : initial value of rho; (default rho=sqrt(n))  
%        pars.update:0  -- fix rho (default); 1 -- update rho during process       
%        pars.LOWBD: upper bound i.e., L=pars.LOWBD.^2, Z_{ij}>=L_{ij}^2 
%        pars.UPPBD: lower bound i.e., U=pars.LOWBD.^2, Z_{ij}<=U_{ij}^2
%                    Note: elements of pars.LOWBD and pars.UPPBD are UNSQUARED distances                          
%        pars.range: the communication range of two points, which means
%                    upper bound for Z_{ij}<=pars.range^2 if (D_{ij}>0  & i~=j)
%                    lower bound for Z_{ij}>=pars.range^2 if (D_{ij}==0 & i~=j)
%                    Note: pars.range is particular for SNL problem. If pars.range
%                    exists, no need pars.LOWBD and pars.UPPBD
%        pars.Otol : tolerance for objective, default Otol=sqrt(n)*1e-3 
%        pars.Etol : tolerance for eigenvalue, default Etol=1e-3  
%                    Note: os the noise is realtely large change Etol=1e-2
%        pars.draw : 1--plot localizations in Re^dim (default); 0--no plot 
%
%
% OUTPUTS:
%
% If NO pars.PP exists 
%       Out.X:     dim-by-n matrix,  final coordinates 
%       Out.Time:  total time
%       Out.stress:relative stress
%
% If pars.PP exists 
%       Out.X:     dim-by-(n-m) matrix, coordinates before refinement 
%       Out.rX:    dim-by-(n-m) matrix, coordinates after refinement 
%       Out.Time:  total time including time for refinement
%       Out.rTime: time for refinement
%       Out.RMSD:  Root Mean Square Distance (RMSD) before refinement 
%       Out.rRMSD: Root Mean Square Distance (RMSD) after refinement 
%
% Refinement step is taken from Kim-Chaun Toh SNLSDP solver
%
% Send your comments and suggestions to   [ sz3g14@soton.ac.uk ]                       
%
% Warning: Accuracy may not be guaranteed!!!!!   
%
% This version: March 1st, 2018,   written by Shenglong Zhou   

fid = fopen('Output.txt','a+');
while fid ==-1
            fid=fopen('Output.txt','a+');
end
fprintf(fid,'**************************************************************\n');
fprintf(fid,'                    SQREDM_SMDS                               \n');
fprintf(fid,'**************************************************************\n');
fclose(fid);
t0=tic;

% parameters design
n  = size(D,1);
if nargin==2; pars=[]; end
[m,itmax,Eigtol,Objtol] = getparameters(n,pars);

fid = fopen('Output.txt','a+');
while fid ==-1
            fid=fopen('Output.txt','a+');
end
if m>0; 
    fprintf(fid,'Number of given anchor points : %3d\n',m);
    fprintf(fid,'Number of unknown points: %3d\n',n-m);
    fprintf(fid,'Procru stes analysis and refinements step will be done!\n');
else
    fprintf(fid,'No anchor points are given!\n'); 
    fprintf(fid,'Number of training points: %3d\n',n);
end
fclose(fid);

Do    = D;
D  = full(D);
% scale = max(D(:));
% if scale<=10; scale=1; else D=D./scale; end
scale=1;


% construction of weight matrix
H  = D ;
r  = dim;
T  = [];
if m>0; T = 1:m; H(T,T)=0;  end

% H2=H2_Gaussian(H,q);
% H2 =  ones(n,1)*(std(H).^(q)); 
H2    = H.^q; 
H2(H==0)=0;
H2(H2==inf)=0;


% construction of supervised dissimilarity matrix
%beta = 2;
%DY_beta = (ones(size(DY,1),1)*beta).^(-1).*DY;
%D_super =((1-alpha)*D+(alpha)*DY_beta.*D)/(1-alpha);
D_super =((1-alpha)*D+(alpha)*DY.*D)/(1-alpha);
 Z  = D_super.*D_super; 
% Z=D.*D;


% construction of bounds
UB = n*max(Z(:));
L  = zeros(n); 
U  = UB*ones(n);

if isfield(pars,'LOWBD');  
    L(DY==1)  = (pars.LOWBD/scale).^2;  
end
if isfield(pars,'UPPBD');  
    U(DY==0)  = (pars.UPPBD/scale).^2;
    HU = spones(tril(U,-1));
    if nnz(HU)<(n^2-n)/2; U=U+(1-HU-HU')*UB; end   % if U_{ij}=0 set U_{ij}=UB
    U(U==inf)=UB;
end 
if isfield(pars,'range'); 
    if pars.range>0
    L(D==0)=(pars.range/scale)^2;                          
    U(D~=0)=(pars.range/scale)^2;
    end
end
L(T,T)       = Z(T,T);      
U(T,T)       = Z(T,T);
% L(DY==1)     = max(max(Z(DY==1)));
L(1:n+1:end) = 0;               
U(1:n+1:end) = 0;  
Z      = min( max(Z, L ), U );


% initailization of rho
% rho    = sqrt(n);
rho    = 0.5;
update = 1; 
if isfield(pars,'rho');    rho    = pars.rho;    end
if isfield(pars,'update'); update = pars.update; end 
 

H2D   = H2.*D_super;  
H2r   = H2/rho;
H2Dr  = H2D/rho;
TH    = find(H2Dr>0);  
PZ    = ProjKr(-Z,r);

fid = fopen('Output.txt','a+');
while fid ==-1
            fid=fopen('Output.txt','a+');
end
fprintf(fid,'Start to run ... \n');
fprintf(fid,'------------------------------------------------\n');
ErrEig  = Inf; 
%FZr     = FNorm(H(TH).*(sqrt(Z(TH))-D_super(TH)))+rho*FNorm(Z+PZ)/2;
FZr     = norm((sqrt(H2(TH)).*(sqrt(Z(TH))-D_super(TH))),'fro')+rho*FNorm(Z+PZ)/2;
iter= 1;
ErrEig0=0;
ErrObj=0;
while (iter<itmax & ~((ErrEig<Eigtol | abs(ErrEig0-ErrEig)<1e-8) & ...
       ErrObj<Objtol & iter>9));
    
   
    Z   = min( max( Lhalf(-PZ-H2r,H2Dr,TH), L ), U );
    
    PZ  = ProjKr(-Z,r);
    
    
    % stop criteria    
    ErrEig0 = ErrEig;
    gZ      = FNorm(Z+PZ)/2;
    ErrEig  = 2*gZ/FNorm(JXJ(Z));
    FZro    = FZr;
    
    FZr     = norm((sqrt(H2(TH)).*(sqrt(Z(TH))-D_super(TH))),'fro')+rho*gZ;
    
    
    ErrObj  = abs(FZro-FZr)/(rho+FZro);
    
    fprintf(fid,'Iter: %3d  ErrEig: %.3e  ErrObj: %.3e\n',iter, ErrEig, ErrObj);
    
    % update rho if update==1
    if update==1 & mod(iter,10)==0
       
        rho0 = rho;
        
        if ErrEig>Eigtol & ErrObj<Objtol/10; rho= 1.1*rho; end
       
        if ErrEig<Eigtol/10 & ErrObj>Objtol; rho= 0.9*rho; end  
       
        if rho0~=rho; H2r = H2/rho; H2Dr  = H2D/rho;       end
        
    end
    
     iter=iter+1;
    
 
end


Out.H2     = H2(1,:);
Out.Time   = toc(t0);
% fprintf('[U,E]      = eig(JXJ(-Z)/2); \n');
% tt=tic;
[U,E]      = eig(JXJ(-Z)/2);
% toc(tt)
Eig        = sort(real(diag(E)),'descend');
Er         = real((Eig(1:r)).^0.5); 
Out.Eigs   = Eig*(scale^2);
Out.X      = (sparse(diag(Er))*real(U(:,1:r))')*scale;
Z          = sqrt(Z)*scale;
Out.stress = sqrt(FNorm(sqrt(Z(H~=0))-Do(H~=0))/FNorm(Do(H~=0)));
Out.alpha  = alpha;
Out.dim    = dim;
Out.q      = q;
fprintf(fid, '------------------------------------------------\n');
fprintf(fid,'Time:      %1.3fsec\n',  Out.Time);
fprintf(fid,'Stress:    %1.2e \n',    Out.stress) ;   
fprintf(fid, '------------------------------------------------\n');

fprintf(fid,'**************************************************************\n\n');
fclose(fid);

end

% ------------------------------------------------------------------------
function  [m,itmax,Eigtol,Objtol] = getparameters(n,pars)
    itmax  = 2000; 
    m      = 0;
    Objtol = sqrt(n)*1e-2;
    Eigtol = 1e-2;              % Eigtol = 1e-2 if the noise factor nf>0.2 
    if isfield(pars, 'Otol');   Objtol=pars.Otol;  end
    if isfield(pars, 'm');      m=pars.m;          end
    if isfield(pars, 'Etol');   Eigtol=pars.Etol;  end
end
 
% ------------------------------------------------------------------------
function  JXJ = JXJ( X )
% Compute J*X*J where J = I-ee'/n;
    nX   = size(X,1);
    Xe   = sum(X, 2);  
    eXe  = sum(Xe);    
    JXJ  = repmat(Xe,1,nX);
    JXJt = repmat(Xe',nX,1);
    JXJ  = -(JXJ + JXJt)/nX;
    JXJ  = JXJ + X + eXe/nX^2;
end

% ------------------------------------------------------------------------
function x =Lhalf( w, a, In )
% min 0.5*(x-w)^2-2*a*x^(0.5) s.t. x>=0. (where a>=0)
[mw,nw]    = size(w);
if nnz(In)/mw/nw < 0.4
	x      = w; 
    a2     = zeros(mw,nw); 
    w3     = zeros(mw,nw) ;
    d      = zeros(mw,nw) ;
    a2(In) = a(In).^2/4; 
    w3(In) = w(In).^3/27; 
    d(In)  = a2(In)-w3(In);

    I1=In(find(d(In)<0));
    if ~isempty(I1) 
        x(I1)=(2*w(I1)/3).*(1+cos((2/3)*acos(sqrt(a2(I1)./w3(I1)))));
    end
    
    I2=In(find( d(In)>=0 & w(In)>=0));
    if ~isempty(I2)
        st2=sqrt(d(I2));
        x(I2)= ((a(I2)/2+st2).^(1/3)+(a(I2)/2-st2).^(1/3)).^2;
    end
    
    I3=In(find( d(In)>=0 & w(In)<0));
    if ~isempty(I3)
        st3=sqrt(d(I3));
        x(I3)= ((a(I3)/2+st3).^(1/3)-(st3-a(I3)/2).^(1/3)).^2;
    end
else
    x=zeros(mw,nw);
    a2=a.^2/4; w3=w.^3/27; d=a2-w3;
    I1=find(d<0); 
    if ~isempty(I1) 
        x(I1)=(2*w(I1)/3).*(1+cos((2/3)*acos(sqrt(a2(I1)./w3(I1)))));
    end
    
    I2=find( d>=0 & w>=0);
    if ~isempty(I2)
        st2=sqrt(d(I2));
        x(I2)= ((a(I2)/2+st2).^(1/3)+(a(I2)/2-st2).^(1/3)).^2;
    end
    
    I3=find( d>=0 & w<0);
    if ~isempty(I3)
        st3=sqrt(d(I3));
        x(I3)= ((a(I3)/2+st3).^(1/3)-(st3-a(I3)/2).^(1/3)).^2;
    end
 
end
x = real(x);
end
% ------------------------------------------------------------------------
function fn= FNorm(A)
% Compute the Frobenius norm of A, i.e., ||A||_F^2
    fn=sum(sum(A.*A));
end

function ip= InnerPro(A,B)
% Compute the inner product of A, i.e., ||A||_F^2
    ip=sum(sum(A.*B));
end

% ------------------------------------------------------------------------
function Z0= ProjKr(A,r)
% The projection of A on cone K_+^n(r)  
	JAJ     = JXJ(A);
	[V0,P0]= eigs((JAJ+JAJ')/2,r,'LA');
	Z0     = real(V0*max(0,P0)*V0'+A-JAJ); 
end

function D_full =  complete_missing_elements(D)
Do    = D;
nD    = nnz(D);% nonzero element of dissimilarity matrix
rate  = nD/n/n;% nonzero rate

% shortest path to complete the missing elements of D
fid = fopen('Output.txt','a+');
while fid ==-1
            fid=fopen('Output.txt','a+');
end
fprintf(fid,'Available dissimilarities rate: %1.2f \n',rate);
fclose(fid);
if  rate<0.05
    fprintf(fid,'Suggest providing more available dissimilarities!\n');
end
if rate<0.9
    D2 = D.*D;    
    fprintf(fid,'Contruct the shortest paths...');
    ts = tic;
    SD = graphallshortestpaths(sparse(sqrt(D2)));
    fprintf(fid,'Done the shortest paths by using %1.2f seconds\n',toc(ts));
    if any(SD == Inf)
    fprintf(fid,'The neighborhood graph is not connected, increase range!');
    end
    SD = max(SD, D);  % to only replace those missing distances  
else
    SD = D;
    fprintf(fid,'No shortest paths calculated!\n');
end
end

function H2=H2_Gaussian(D,q)
H2=zeros( size(D,1), size(D,2));
for i = 1:size(D,1)
    for j = 1:size(D,2)
        H2(i,j)= exp(D(i,j)^2/(q*max(max(D(i,:)),max(D(:,j)))^2));
    end
end
end



