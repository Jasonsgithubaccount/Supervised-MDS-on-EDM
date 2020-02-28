function F_crit= evaltestcrit(D_te, Out, X_new, DY_mod, alpha, H2)
    
    n   = size(DY_mod,2); 
    %n_1 = size(PY(PY(1,:)==1));
    %n_2 = size(PY(PY(1,:)==-1));
    %t   = size(par.PS_te,2);
    % Case 1: y_new
    z_z_new = Out.X-X_new*ones(1,size(Out.X,2));
    multi_z_z_new_2 = (Out.X-X_new*ones(1,size(Out.X,2))).^2;
    norm_multi_z_z_new_2 = sum(((Out.X-X_new*ones(1,size(Out.X,2))).^2),1);
    norm_multi_z_z_new = sqrt(sum(((Out.X-X_new*ones(1,size(Out.X,2))).^2),1));
    
    
    FSum = sum(((D_te - sqrt(sum(((Out.X-X_new*ones(1,size(Out.X,2))).^2),1))).^2).*H2,2);
    FSqSum = sum(-2*DY_mod.* D_te.*H2.*sqrt(sum(((Out.X-X_new*ones(1,size(Out.X,2))).^2),1)),2);
    
    F_crit = (1-alpha)*FSum + alpha*FSqSum;
    
end