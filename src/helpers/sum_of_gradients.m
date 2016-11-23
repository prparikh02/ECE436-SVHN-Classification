function G = sum_of_gradients(A)

G = zeros(size(A));
m = 32;
n = 32;

for k = 1:size(A,2)
    [Gx, Gy] = imgradient(reshape(A(:,k),m,n));
    G(:,k) = reshape(Gx+Gy,m*n,1);
end