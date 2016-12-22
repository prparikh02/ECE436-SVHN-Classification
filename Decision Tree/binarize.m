function [D, thresh] = binarize(D)
% BINARIZE Convert image from grayscale to binary using Otsu's method
%   I = BINARIZE(D)
%   [I, thresh] = BINARIZE(D) also returns calculated grayscale threshold
%   per column

N = size(D, 2);
thresh = zeros(1, N);
parfor k = 1:N
    d = D(:, k);
    thresh(k) = graythresh(d);
    D(:, k) = im2bw(d, thresh(k));
end
