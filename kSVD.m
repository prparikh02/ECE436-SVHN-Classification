function [D,x,Err] = kSVD(Y,D,T0,Td,tol)

D = normc(D);
K = size(D,2);
Err = zeros(1, Td);

for iter = 1:Td
    tic
    % SparseCodingStage
    x = full(omp(D,Y,D'*D,T0));
    
    % Dictionary Update
    for k = 1:K
     
        wk = find(x(k,:));
        Ek = (Y-D*x) + D(:,k)*x(k,:);
        ERk = Ek(:,wk);
        
        if ~isempty(wk)
            [U,S,V] = svds(ERk,1);
            D(:,k) = normc(U);
            x(k,wk) = S*V';
        end
        
        if mod(k,100) == 0
            fprintf('iters: %d | k: %d \n',[iter,k]');
        end
        
    end
    Err(iter) = norm(Y-D*x, 'fro');
    fprintf('Error: %f \n',Err(iter));
    if Err(iter) < tol
        break;
    end
    toc
end

Err = Err(:,1:iter);  % Trim trailing zero columns in the event of early break
end