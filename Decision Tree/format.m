% load a specified data file
clear, clc, close all
load('train_28x28.mat')
%%

% Generate 28x28 images
[x_,y_,~,n] = size(X);
p = 28*28;

data = uint8(zeros(n,p));
parfor i=1:n
    data(i,:) = reshape(rgb2gray(imresize(X(:,:,:,i), [28 28])),1,p);
end
%%

% Generate cropped images (remove 4 columns on each side), then resize
[x_,y_] = size(data);
p = 20*28;

X = uint8(zeros(x_,p));
parfor i=1:x_
    im = reshape(data(i,:), 28, 28);
    X(i,:) = uint8(reshape(im(:, 5:24), 1, 560));
end

data = X;
%%

% Binarize images as uint8 matrices with 0 = black, 255 = white based on
% recommendations from Otsu's method
[x_,y_] = size(data);
p = x_*y_;

X = uint8(zeros(x_,y_));
b = binarize(data);
for i=1:x_
    for j=1:y_
        if b(i,j) == logical(1)
            X(i,j) = uint8(255);
        else
            X(i,j) = uint8(0);
        end
    end
end

data = X;
