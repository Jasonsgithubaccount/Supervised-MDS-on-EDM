function D2 = GDTW_dist(ZI, ZJ)
    m2 = size(ZJ,1);
    %n  = size(ZI,2);
    D2 = zeros(1,m2);
    for i = 1:m2
        D2(i) = exp(-dtw(ZI,ZJ(i,:))/2);
    end
end