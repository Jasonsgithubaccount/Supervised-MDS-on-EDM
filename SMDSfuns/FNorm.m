function fn= FNorm(A)
% Compute the Frobenius norm of A, i.e., ||A||_F^2
    fn=sum(sum(A.*A));
end