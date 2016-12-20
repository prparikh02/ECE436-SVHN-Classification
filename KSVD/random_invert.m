function D = random_invert(D, varargin)
% RANDOM_INVERT Randomly invert columns of D
%   D = RANDOM_INVERT(D)
%   D = RANDOM_INVERT(D, p) specifiy probability of invert, otherwise 0.50

if nargin == 1
    p = 0.50;
elseif nargin == 2
    p = varargin{1};
else
    error('too many arguments');
end

N = size(D, 2);
a = rand(1, N) < p;
k = find(a==1);

D(:, k) = imcomplement(D(:, k));