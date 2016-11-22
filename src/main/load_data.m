function [D, D_labels, Y, Y_labels] = load_data(digits, K, N)
% LOAD_DATA Load K training samples and N testing samples of each digit in c
%   [D, D_labels, Y, Y_Labels] = LOAD_DATA(digits, K, N)

load('train_32x32.mat');

[m,n,~,s] = size(X);
D = uint8(zeros(m*n,s));
D_labels = y;

for k = 1:s
    D(:,k) = reshape(rgb2gray(X(:,:,:,k)),m*n,1);
end

clearvars m n s k X y;

load('test_32x32.mat');

[m,n,~,s] = size(X);
Y = uint8(zeros(m*n,s));
Y_labels = y;

for k = 1:s
    Y(:,k) = reshape(rgb2gray(X(:,:,:,k)),m*n,1);
end

clearvars m n s k X y;

digits(digits==0) = 10; % zeros are actually mapped to 10 in labels
num_classes = length(digits);

rnd_ordering = randperm(size(D,2));
counts = zeros(1,10); % could probably do better using unique(digits)
DL = zeros(1,K*num_classes);
dl_i = 1;
for k = rnd_ordering
    d = digits(find(D_labels(k) == digits));
    if ~isempty(d) && counts(d) < K
        DL(dl_i) = k;
        dl_i = dl_i + 1;
        counts(d) = counts(d) + 1;
    end
    
    if sum(counts) == K*num_classes
        break;
    end
end
D = D(:,DL);
[D_labels, idx] = sort(D_labels(DL));
D = D(:,idx);

rnd_ordering = randperm(size(Y,2));
counts = zeros(1,num_classes);
YL = zeros(1,N*num_classes);
yl_i = 1;
for k = rnd_ordering
    d = find(Y_labels(k) == digits);
    if ~isempty(d) && counts(d) < N
        YL(yl_i) = k;
        yl_i = yl_i + 1;
        counts(d) = counts(d) + 1;
    end
    
    if sum(counts) == N*num_classes
        break;
    end
end

Y = Y(:,YL);
[Y_labels, idx] = sort(Y_labels(YL));
Y = Y(:, idx);