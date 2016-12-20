%% Trial Runs

rng(10);

resize = [false, true];
bin = [false, true];
crop = [false, true];
% rand_invert = [false, true];
% block_sampling = false;
% tol = 1e-3;

digits = [1, 4, 5, 7];
K = 500;
N = 100;
Td = 10;
T0 = 10;

C = {};
Err = {};
RT = {};
k = 1;
for r = resize
    for b = bin
        for c = crop
            tic;
            [C{k}, Err{k}, D] = ...
                svhn_ksvd_classifier(digits, K, N, T0, Td, r, b, c);
            RT{k} = toc;
            k = k + 1;
            close all;
        end
    end
end

save('results2.mat', 'C', 'Err', 'RT')
