function [X, cnt] = createTensorOrder(order)
    X = nchoosek(kron([1, 2, 3], ones(1, order)), order);
    X = unique(X, 'rows');
    for i = 1:size(X, 1)
        cnt(i) = factorial(order) / factorial(nnz(X(i, :) ==1))/ factorial(nnz(X(i, :) ==2))/ factorial(nnz(X(i, :) ==3));
    end

end