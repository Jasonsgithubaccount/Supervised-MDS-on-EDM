function  [m,itmax,Eigtol,Objtol] = getparameters(n,pars)
    itmax  = 2000; 
    m      = 0;
    Objtol = sqrt(n)*1e-2;
    Eigtol = 1e-3;              % Eigtol = 1e-2 if the noise factor nf>0.2 
    if isfield(pars, 'Otol');   Objtol=pars.Otol;  end
    if isfield(pars, 'm');      m=pars.m;          end
    if isfield(pars, 'Etol');   Eigtol=pars.Etol;  end
end
