function [D, V] = binarize(D, c)
% BINARIZE Convert image to binary
%   I = BINARIZE(D, c) returns binary version of D corresponding to
%   cutoff threshold, c
%   [I, V] = BINARIZE(D, c) also returns calculated cutoff value, V

% default c
if c <= 0 || c > 1.0
    c = 0.50;
end

s = sort(D(:));
V = s(floor(c*numel(D)));

D(D<V) = 0;
D(D>=V) = 255;
% D = logical(D);