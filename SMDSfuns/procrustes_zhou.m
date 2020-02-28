function Xs= procrustes_zhou(m0, PP, D0)
% Input:
% m0 -- the number of given anchors in Re^r0
% PP -- r0-by-n0 matrix whose first m0 columns are  given anchors 
%       and rest (n0-m0) columns are given sensors
% D0--  n0-by-n0 distance matrix 
%      [anchors-anchors (squred)distance, anchors-sensors (squred)distance
%       sensors-anchors (squred)distance, sensors-sensors (squred)distance]
%       containing m0 computed anchors and (n0-m0) sensors in Re^r0
% Output:
% Xs -- r0-by-(n0-m0) matrix containing (n0-m0) sensors in Re^r0
    r0     = size(PP,1);
    n0     = size(D0,2);
    JDJ    = JXJ(D0);
    JDJ    = -(JDJ+JDJ')/4;
    [U1,D1]= eigs(JDJ,r0,'LA');   
    X0     = (D1.^(1/2))*(U1');   
    if m0>0
    A      = PP(:,1:m0);
    [Q,~,a0,p0] = procrustes_qi(A,X0(:,1:m0));	
	Z0     = Q'*(X0-p0(:, ones(n0, 1))) + a0(:, ones(n0, 1)); 
    Xs     = Z0(:,m0+1:n0);
    Xa     = Z0(:, 1:m0);
    %Xs    = Xs*mean(sum(Xa.*A)./sum(Xa.^2));
    Xs     = Xs*max(1,sum(sum(Xa.*A))/sum(sum(Xa.^2)));
    else        
    [~,~,Map]= procrustes(PP',X0'); 
    Xs       = (Map.b*Map.T')*X0 + diag(Map.c(1,:))*ones(size(X0));
    end
end

% ------------------------------------------------------------------------
function [Q, P, a0, p0] = procrustes_qi(A, P)
% Procrustes analysis for rigid localization of anchors in A
% Input:
% A -- r-by-m0 matrix containing m0 anchors in Re^r
% P -- r-by-m0 matrix containing m0 computed locations of m0 anchors in Re^r
% 
% Output:
% Q  -- the orthogonal transformation
% P  -- the resulting coordinators of P after transformation
% a0 -- the tranlation vector that simply centrizes A
% p0 -- the tranlation vector that simply centrizes P
%
%
    m0 = size(A,2);
    % centerizing A and P
    a0 = sum(A, 2)/m0;
    p0 = sum(P,2)/m0;
    A  = A - a0(:, ones(m0,1));
    P1 = P - p0(:, ones(m0,1));
    P  = P1*A';
    [U0, ~, V] = svd(P);
    Q  = U0*V';
    P  = Q'*P1 + a0(:, ones(m0,1));
end