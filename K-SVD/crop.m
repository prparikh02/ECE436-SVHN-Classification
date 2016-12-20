function D = crop(D, rect)
% CROP removes the top, bottom, left, and right columns as specified by the
%   length-4 vector input, RECT: [TOP, BOTTOM, LEFT, RIGHT]. Input D is
%   the entire dictionary.
%   D = CROP(D)

T = rect(1);
B = rect(2);
L = rect(3);
R = rect(4);

d = size(D, 1);
m = sqrt(d);
n = sqrt(d);

D = D(L*m+1:end-R*m, :);
% D(L*m+1:end-R*m, :) = max(D(:));
% D = D';
% D = D(T*n+1:end-B*n, :);
% D = D';