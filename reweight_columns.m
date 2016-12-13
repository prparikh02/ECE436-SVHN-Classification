function D = reweight_columns(D)
% REWEIGHT_COLUMNS Weight each column by fading out the outer columns and
% keeping the origanl intensity of the middle columns
%   D = REWEIGHT_COLUMNS(D)

[d, n] = size(D);
x = linspace(-1, 1, n);

w = normpdf(x);
w = repmat(w/max(w), d, 1);
% w = repmat(w(:), 1, n);

D = D.*uint8(w);
