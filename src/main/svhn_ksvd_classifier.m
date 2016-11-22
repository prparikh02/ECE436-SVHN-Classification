function [C, Err] = svhn_ksvd_classifier(digits, K, N, T0, Td, varargin)
%% Parth Parikh

%% Argument check

if nargin == 6
    tol = varargin{1};
elseif nargin > 6
    error('too many arguments');
else
    tol = 1e-5;
end

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
[D, D_labels, Y, Y_labels] = load_data(digits, K, N);

% D = double(binarize(D, 0.20));
% Y = double(binarize(Y, 0.20));

D = double(D);
Y = double(Y);

%%

fprintf('\n----------Beginning KSVD----------\n');

% T0 = 3;
% Td = 50;

[D_learned, x, Err] = kSVD(Y, D, T0, Td, tol);

%%

fprintf('\n----------Calculating Results----------\n');

splits = 1:K:size(D_learned,2);
num_digits = length(digits);
num_signals = size(Y,2);
res = zeros(1,num_signals);
c_star = zeros(1,num_signals);
for s = 1:length(res)
    ys = double(Y(:,s));
    r = zeros(1,num_digits);
    for c = 1:num_digits
        Dc = D_learned(:,splits(c):splits(c)+K-1);
        xc = x(splits(c):splits(c)+K-1,:);
        r(c) = norm(ys - Dc*xc(:,s),2);
    end
    [~,ind] = min(r);
    c_star(s) = digits(ind);
end

s = 1:num_signals;

figure;
plot(1:Td, Err, 'r--*');

splits = 1:N:size(Y,2);
accuracy = zeros(1, num_digits);
for k = 1:num_digits
    accuracy(k) = sum((Y_labels(splits(k):splits(k)+N-1)==c_star(splits(k):splits(k)+N-1)'))/N;
end

fprintf('\n digit | accuracy \n');
fprintf('-------|----------\n');
fprintf('  %2d   | %5.4f  \n',[digits',accuracy']');
fprintf('\nTotal Accuracy: %5.4f\n',mean(accuracy));

C = confusionmat(Y_labels,c_star);
fprintf('\nConfusion Matrix \n  |');
fprintf('%6d ', digits);
fprintf('\n--|------------------------------------\n');
for k = 1:num_digits
    fprintf('%d | ',digits(k));
    fprintf('%4.3f  ',C(k,:)/N);
    fprintf('\n');
end
fprintf('\n');