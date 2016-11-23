function [D, D_labels, Y, Y_labels] = load_data(digits, K, N, resize)
% LOAD_DATA Load K training samples and N testing samples of each digit in c
%   [D, D_labels, Y, Y_Labels] = LOAD_DATA(digits, K, N)

load('train_32x32.mat');

scale = 0.75;

[m,n,~,s] = size(X);
if resize
    D = uint8(zeros(m*n*scale^2,s));
else
    D = uint8(zeros(m*n,s));
end
D_labels = y;

parfor k = 1:s
    d = rgb2gray(X(:,:,:,k));
    if resize
        d = imresize(d, scale);
    end
    D(:,k) = d(:);
end

clearvars m n s k X y;

load('test_32x32.mat');

[m,n,~,s] = size(X);
if resize
    Y = uint8(zeros(m*n*scale^2,s));
else
    Y = uint8(zeros(m*n,s));
end
Y_labels = y;

parfor k = 1:s
    y = rgb2gray(X(:,:,:,k));
    if resize
        y = imresize(y, scale);
    end
    Y(:,k) = y(:);
end

clearvars m n s k X y;

num_classes = length(digits);

%%
rng(42)
%%
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