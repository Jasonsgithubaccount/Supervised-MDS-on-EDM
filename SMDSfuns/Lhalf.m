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