function D_re = resize(D, dims)
%RESIZE Similar to IMCROP, but for entire Dictionary
%   dims = [NUM_ROWS, NUM_COLS]

N = size(D, 2);
newL = prod(dims);
D_re = uint8(zeros(newL, N));

parfor k=1:N
    im = reshape(D(:,k), 32, 32);
    im_re = imresize(im, dims);
    D_re(:,k) = im_re(:);
end
