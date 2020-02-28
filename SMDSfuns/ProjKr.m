function Z0= ProjKr(A,r)
% The projection of A on cone K_+^n(r)  
	JAJ     = JXJ(A);
	[V0,P0]= eigs(real(JAJ+JAJ')/2,r,'LA');
	Z0     = real(V0*max(0,P0)*V0'+A-JAJ); 
end