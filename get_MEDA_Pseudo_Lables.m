function [Acc,acc_iter,Beta,Yt_pred] = get_MEDA_Pseudo_Lables(iter,Xs,Ys,Xt,Yt,options)
%% Inputs:
%%% Xs      : Source domain feature matrix, n * dim
%%% Ys      : Source domain label matrix, n * 1
%%% Xt      : Target domain feature matrix, m * dim
%%% Yt      : Target domain label matrix, m * 1 (only used for testing accuracy)
%%% options : algorithm options:
%%%%% options.d      :  dimension after manifold feature learning (default: 20)
%%%%% options.T      :  number of iteration (default: 10)
%%%%% options.lambda :  lambda in the paper (default: 10)
%%%%% options.eta    :  eta in the paper (default: 0.1)
%%%%% options.rho    :  rho in the paper (default: 1.0)
%%%%% options.base   :  base classifier for soft labels (default: NN)

%% Outputs:
%%%% Acc      :  Final accuracy value
%%%% acc_iter :  Accuracy value list of all iterations, T * 1
%%%% Beta     :  Cofficient matrix
%%%% Yt_pred  :  Prediction labels for target domain
    
    %% Load algorithm options
    if ~isfield(options,'p')
        options.p = 10;
    end
    if ~isfield(options,'eta')
        options.eta = 0.1;
    end
    if ~isfield(options,'lambda')
        options.lambda = 1.0;
    end
    if ~isfield(options,'rho')
        options.rho = 1.0;
    end
    if ~isfield(options,'T')
        options.T = 1;
    end
    if ~isfield(options,'d')
        options.d = 20;
    end


    X = [Xs,Xt];
    n = size(Xs,2);
    m = size(Xt,2);
    nst = size(X,2);
    class = unique(Ys);
    C = length(unique(Ys));
    acc_iter = [];
    
    YY = [];
    for c = 1 : C
        YY = [YY,Ys==c];
    end
    YY = [YY;zeros(m,C)];

    %% Data normalization
    X = X * diag(sparse(1 ./ sqrt(sum(X.^2))));

    % Construct kernel
    g = sqrt(sum(sum(X .^ 2).^0.5)/(n + m));
    K = kernel_meda('rbf',X,g);
    E = diag(sparse([ones(n,1);zeros(m,1)]));

    for t = 1 : options.T
        % Compute coefficients vector Beta
        Beta = ((E ) * K + options.eta * speye(n + m,n + m)) \ (E * YY);
        F = K * Beta;
        [~,Cls] = max(F,[],2);

        %% Compute accuracy
        Acc = numel(find(Cls(n+1:end)==Yt)) / m;
        Cls = Cls(n+1:end);
        acc_iter = [acc_iter;Acc];
        fprintf('Iteration:[%02d]>>,Acc=%f\n',iter,Acc);
    end
    Yt_pred = Cls;
end

function K = kernel_meda(ker,X,sigma)
    switch ker
        case 'linear'
            K = X' * X;
        case 'rbf'
            n1sq = sum(X.^2,1);
            n1 = size(X,2);
            D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
            K = exp(-D/(2*sigma^2));        
        case 'sam'
            D = X'*X;
            K = exp(-acos(D).^2/(2*sigma^2));
        otherwise
            error(['Unsupported kernel ' ker])
    end
end