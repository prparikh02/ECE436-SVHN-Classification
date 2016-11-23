function [C, Err] = svhn_ksvd_classifier(digits, K, N, T0, Td, varargin)
%% Parth Parikh

%% Argument check

if nargin == 6
    tol = varargin{1};
    resize = false;
elseif nargin == 7
    tol = varargin{1};
    resize = varargin{2};
elseif nargin > 7
    error('too many arguments');
else
    tol = 1e-5;
    resize = false;
end

digits(digits==0) = 10; % zeros are actually mapped to 10 in labels
digits = sort(digits);

%% Add Paths
addpath('../../SVHN-dataset');
addpath('../helpers');
addpath('../helpers/ompbox10/');
addpath('../helpers/ksvdbox13/');
% addpath('../test');

%% Load Data

fprintf('----------Loading Data----------\n');

% digits = [2, 3, 9];
% K = 20;
% N = 50;

tic
[D, D_labels, Y, Y_labels] = load_data(digits, K, N, resize);
toc

% D = double(binarize(D, 0.20));
% Y = double(binarize(Y, 0.20));
% D = double(sum_of_gradients(D));
% Y = double(sum_of_gradients(Y));

idx = randperm(size(D,2));
D = D(:,idx);
D_labels = D_labels(idx);
D = double(D);
Y = double(Y);

%%

fprintf('\n----------Beginning KSVD----------\n');

% T0 = 3;
% Td = 50;

[D_learned, x, Err] = kSVD(Y, D, T0, Td, tol);

%%

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
%     for c = 1:num_digits
%         Dc = D_learned(:,splits(c):splits(c)+K-1);
%         xc = x(splits(c):splits(c)+K-1,:);
%         r(c) = norm(ys - Dc*xc(:,s),2);
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
    fprintf('%4.3f  ',C(k,:)/N); % TODO: variable number of samples per class
    fprintf('\n');
end
fprintf('\n');