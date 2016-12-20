function [newFeature, D_labels, newFeatureY, Y_labels] = test_load_data(digits, K, N)

%% load data
% addpath('../../SVHN-dataset');
% addpath('../helpers');
% addpath('../helpers/ompbox10/');
% addpath('../helpers/ksvdbox13/');
% N = 250;
[ D, D_labels, y, Y_labels] = load_data(digits, K, N);
%% extract histogram of gradient feature
% close all
[hog, ~] = extractHOGFeatures(reshape(D(:,1), [32 32]), 'CellSize',[4 4]);
% % points = detectMinEigenFeatures(reshape(D(:,5), [32 32]));
% figure
% % imshow(reshape(D(:,3),[32 32]))
% % I = insertMarker(reshape(D(:,3),[32 32]), vis)
% plot(vis)
%% batch processing
hogFeatureSize = length(hog);
newFeature = zeros(hogFeatureSize ,size(D,2), 'single');
parfor i = 1:size(D,2)
    img = reshape(D(:,i),[32 32]);
%     level = graythresh(img);
%     img = im2bw(img, level);
    img = imbinarize(img, 'adaptive');
    newFeature(:,i) = extractHOGFeatures(img, 'CellSize', [4 4]);
end
%%
newFeatureY = zeros(hogFeatureSize,size(y,2), 'single');
parfor i = 1:size(y,2)
    img = reshape(y(:,i),[32 32]);
    img = imbinarize(img,'adaptive');
    newFeatureY(:, i) = extractHOGFeatures(img, 'CellSize',[4 4]);
end
