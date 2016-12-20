function s = block_sample(x)

x = reshape(x, 32, 32);
[rx, cx] = size(x);
R = 1:2:rx;
C = 1:2:cx;

rs = length(R);
cs = length(C);

s = zeros(rs*cs, 1);
k = 1;
for c = C
    for r = R
        block = x(r:r+1, c:c+1);
        s(k) = mean(block(:));
        k = k + 1;
    end
end

max_s = max(s(:));
s = 255*s/max_s;

% figure;
% subplot(1,2,1);
% imshow(uint8(a));
% subplot(1,2,2);
% imshow(uint8(s));