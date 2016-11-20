%% Parth Parikh

%% Add Paths
addpath('../../SVHN-dataset');
addpath('../helpers');
addpath('../test');

%% Load Data
clear, close all, clc;

load('train_32x32.mat');

[m,n,~,N] = size(X);
D_train = uint8(zeros(m*n,N));

for k = 1:N
    D_train(:,k) = reshape(rgb2gray(X(:,:,:,k)),m*n,1);
end

clearvars m n k X; 

%% 

binarize(D_train(:,112))