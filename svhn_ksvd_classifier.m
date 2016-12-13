function [C, Err, D_learned] = svhn_ksvd_classifier(digits, K, N, T0, Td, varargin)
%% Parth Parikh

%% Argument check

% default params
bin = false;
crop = false;
rand_invert = false;
block_sampling = false;
tol = 1e-5;

% TODO: make this a struct
if nargin == 6
    bin = varargin{1};
elseif nargin == 7
    bin = varargin{1};
    crop = varargin{2};
elseif nargin == 8
    bin = varargin{1};
    crop = varargin{2};
    rand_invert = varargin{3};
elseif nargin == 9
    bin = varargin{1};
    crop = varargin{2};
    rand_invert = varargin{3};
    block_sampling = varargin{4};
elseif nargin == 10
    bin = varargin{1};
    crop = varargin{2};
    rand_invert = varargin{3};
    block_sampling = varargin{4};
    tol = varargin{5};
else
    error('too many arguments');
end

digits(digits==0) = 10; % zeros are actually mapped to 10 in labels
digits = sort(digits);

%% Add Paths
addpath('./ompbox10/');
addpath('./ksvdbox13/');

%% Load Data

fprintf('----------Loading Data----------\n');

tic
[D, D_labels, Y, Y_labels] = load_data(digits, K, N);
toc

%% Preprocessing
if bin
    D = binarize(D);
    Y = binarize(Y);
end

if crop
    D = reweight_columns(D);
    Y = reweight_columns(Y);
end

if rand_invert
    D = random_invert(D, 0.20);
    Y = random_invert(Y, 0.20);
end

if block_sampling
    D = block_sample(D);
    Y = block_sample(Y);
end

% shuffle dictionary
idx = randperm(size(D,2));
D = D(:,idx);
D_labels = D_labels(idx);

% convert to double for computations
D = double(D);
Y = double(Y);

%% K-SVD

fprintf('\n----------Beginning KSVD----------\n');

[D_learned, x, Err] = kSVD(Y, D, T0, Td, tol);

%% Calculate Results

fprintf('\n----------Calculating Results----------\n');

num_digits = length(digits);
num_signals = size(Y,2);
res = zeros(1,num_signals);
c_star = zeros(1,num_signals);
for s = 1:length(res)
    ys = Y(:,s);
    r = zeros(1,num_digits);
    for c = 1:num_digits
        p = find(D_labels==digits(c));
        Dc = D_learned(:,p);
        xc = x(p,:);
        r(c) = norm(ys-Dc*xc(:,s),2);
    end
    [~,ind] = min(r);
    c_star(s) = digits(ind);
end

s = 1:num_signals;

figure;
plot(1:length(Err), Err, 'r--*');

C = confusionmat(Y_labels,c_star);
fprintf('\nConfusion Matrix \n  |');
fprintf('%6d ', digits);
fprintf('\n--|------------------------------------\n');
for k = 1:num_digits
    fprintf('%d | ',digits(k));
    fprintf('%4.3f  ',C(k,:)/N);  % TODO: variable number of samples per class
    fprintf('\n');
end
fprintf('\n');