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