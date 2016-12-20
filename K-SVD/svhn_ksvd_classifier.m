function [C, Err, D_learned] = svhn_ksvd_classifier(digits, K, N, T0, Td, varargin)
%% Parth Parikh

%% Argument check

p = inputParser;
addOptional(p, 'resize', false);
addOptional(p, 'binarize', false);
addOptional(p, 'crop', false);
addOptional(p, 'randomInvert', false);
addParameter(p, 'blockSample', false);
addParameter(p, 'tol', 1e-5, @isnumeric);
parse(p, varargin{:});
p.Results

tol = p.Results.tol;

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

if p.Results.resize
    D = resize(D, [28, 28]);
    Y = resize(Y, [28, 28]);
end

if p.Results.binarize
    D = binarize(D);
    Y = binarize(Y);
end

if p.Results.crop
    D = crop(D, [0, 0, 4, 4]);
    Y = crop(Y, [0, 0, 4, 4]);
end

if p.Results.randomInvert
    D = random_invert(D, 0.20);
    Y = random_invert(Y, 0.20);
end

if p.Results.blockSample
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