function [Z,acc_ite,Y_tar_pseudo,fixlable_info,TheFixLabelYt] =VTL(Xs,Ys,Xt,Yt,options,optionss)
    T = options.T;
    
    inIter=options.inIter;
    acc_ite = [];
    
    %% Iteration
    [H]=tabulate(Ys);
    TheFixLabelYt=zeros(size(Yt));
    
    Xs_ori=Xs;
    Xt_ori=Xt;
    Y_tar_pseudo = [];
    fixlable_info=[];
    label_table=ones(size(Xt,1),size(H,1));
    for i = 1 : T
        label_table=ones(size(Xt,1),size(H,1));
        label_table_layers=[];
%         Y_tar_pseudo = [];
        fprintf('--------------%dth result--------------\n',i);
        for j=1:inIter
            [Z,~] = getProjection(Xs,Ys,Xt,Y_tar_pseudo,options,optionss,label_table);
            Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
            Zs = Z(:,1:size(Xs,1));
            Zt = Z(:,size(Xs,1)+1:end);

            option.T=1;
            option.eta=0.1;
            [acc,~,~,Y_tar_pseudo] = get_MEDA_Pseudo_Lables(j,Zs,Ys,Zt,Yt,option);
            
            [label_table,label_table_layers] = UpdataTable(Y_tar_pseudo,max(Ys),label_table,label_table_layers,j,inIter);
            
            [~,label_table_sort]=sort(label_table,2,'descend');
            [label_table_value,~]=sort(label_table,2,'descend');
            [info_shang] = Calcu_info(label_table);%Calculate information entropy
            
            info_table_sort=[info_shang,label_table_sort(:,1)];
            
            if i>1
                [Y_tar_pseudo] = RepairPreLable(TheFixLabelYt,Y_tar_pseudo);
            end
            
            Xs =Zs';%update data
            Xt =Zt';
            acc_ite = [acc_ite;acc];
        end
        
        [TheFixLabelYt] = UpdateFix_basedInfo(info_shang,TheFixLabelYt,label_table);
        
        fixLength_rateAcc=length(find(TheFixLabelYt==Yt))/length(find(TheFixLabelYt>0));
        fprintf('-------Accuracy of fixed labels: %0.4f  Number of correct fixed labels:%d  The number of all fixed labels:%d  Fixed rate:%0.4f-----------\n',fixLength_rateAcc,length(find(TheFixLabelYt==Yt)),...
            length(find(TheFixLabelYt>0)),length(find(TheFixLabelYt>0))/length(Yt));
        fixlable_info=[fixlable_info;i,fixLength_rateAcc,length(find(TheFixLabelYt==Yt)),length(find(TheFixLabelYt>0)),length(find(TheFixLabelYt>0))/length(Yt)];
        %continue to use the original data
        Xs=Xs_ori;
        Xt=Xt_ori;
    end
    
    fprintf('\n--------------------------maxAcc=%0.4f-------------------------------\n',max(acc_ite));
    
end

function [Z,A] = getProjection(Xs,Ys,Xt,Y_tar_pseudo,options,optionss,label_table)
	%% Set options
	lambda = options.lambda;              %% lambda for the regularization
	dim = options.dim;                    %% dim is the dimension after adaptation, dim <= m
	kernel_type = options.kernel_type;    %% kernel_type is the kernel name, primal|linear|rbf
    
    beta = options.beta;
    mu = options.mu;
    %%DICD options
    alpha = optionss.alpha;
    ru=optionss.ru;
    
	%% Construct MMD matrix
	X = [Xs',Xt'];
    Xs1 = Xs';
    ns1 = size(Xs1,1);
	X = X*diag(sparse(1./sqrt(sum(X.^2))));
	[m,n] = size(X);
	ns = size(Xs,1);
	nt = size(Xt,1);
	e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
	C = length(unique(Ys));

	%%% M0
	M = e * e' * C;  %multiply C for better normalization
    
  %scatter for source domain  
   class = unique(Ys);
   results = [];  

	%%% Mc
	N = 0;
    allQ = zeros(size(Xs,2),size(Xs,2)); 
    
    
    allmeanC=[];
    Yall=[Ys;Y_tar_pseudo];
    Xall=[Xs;Xt];
    
    for i=1:C
        Xi = Xall(find(Yall==class(i)),:);
        meanClass = mean(Xi,1);
        allmeanC=[allmeanC;meanClass];
    end
    
    if ~isempty(Y_tar_pseudo) && length(Y_tar_pseudo)==nt
        for c = reshape(unique(Ys),1,C)
            e = zeros(n,1);
            e(Ys==c) = 1 / length(find(Ys==c));
            e(ns+find(Y_tar_pseudo==c)) = -1 / length(find(Y_tar_pseudo==c));
            e(isinf(e)) = 0;
            N = N + e*e';
        end
        [allQ,R] = Calcu_Q(Xs,Ys,Xt,Y_tar_pseudo,label_table);
        %D same in Sourse
        DsS=zeros(ns,ns);
        for i=1:ns
           nc = length(find(Ys==Ys(i)));
           for j=1:ns
               if i==j
                   DsS(i,j)= ns;
               elseif Ys(i)==Ys(j)
                   DsS(i,j)=-(ns)/nc;
               else
                   DsS(i,j)=0;
               end

            end
        end

        %D same in Target
        DsT=zeros(nt,nt);
        for i=1:nt
           nc = length(find(Y_tar_pseudo==Y_tar_pseudo(i)));
           for j=1:nt
               if i==j
                   DsT(i,j)= nt;
               elseif Y_tar_pseudo(i)==Y_tar_pseudo(j)
                   DsT(i,j)=-(nt)/nc ;  
               else
                   DsT(i,j)= 0;
               end
            end
        end
        
        
        % D smae in Source to Target
        DsST=zeros(ns,nt);
        for i=1:ns
           for j=1:nt
               if Ys(i)==Y_tar_pseudo(j)
                   DsST(i,j)=-(((ns)/length(find(Ys==Ys(i))))*((nt)/length(find(Y_tar_pseudo == Y_tar_pseudo(j)))));
               end
           end
        end
            
            
        % D smae in Target to Source
        DsTS=zeros(nt,ns);
        for i=1:nt
           for j=1:ns
               if Y_tar_pseudo(i)==Ys(j)
                   DsTS(i,j)=-(((nt)/length(find(Y_tar_pseudo == Y_tar_pseudo(i))))*((ns)/length(find(Ys==Ys(j)))));
               end
           end
        end

        
        %D different in Source
        DdS=zeros(ns,ns);
        for i=1:ns
            nc = length(find(Ys==Ys(i)));
           for j=1:ns
               if i==j
                   DdS(i,j)= ns - nc;
               elseif Ys(i)~= Ys(j)
                   DdS(i,j)= -1;
               else
                   DdS(i,j)= 0;
               end
           end
        end

        %D different in Target
        DdT=zeros(nt,nt);
        for i=1:nt
           nc = length(find(Y_tar_pseudo==Y_tar_pseudo(i)));
           for j=1:nt
               if i==j
                   DdT(i,j)= nt - nc;
               elseif Y_tar_pseudo(i)~= Y_tar_pseudo(j)
                   DdT(i,j)= -1;
               else
                  DdT(i,j)= 0;
               end
           end
        end
        
         %D different in Source to Target
        DdST=zeros(ns,nt);
        for i=1:ns
           for j=1:nt
               if  Ys(i)~= Y_tar_pseudo(j)
                   DdST(i,j)= -1;
               else
                   DdST(i,j)=0;
               end
            end
        end
        
        %D different in Target to Source
        DdTS=zeros(nt,ns);
        for i=1:nt
           for j=1:ns
               if  Y_tar_pseudo(i)~= Ys(j)
                   DdTS(i,j)= -1;
               else
                   DdTS(i,j)= 0;
               end
            end
        end
        

        DsS = DsS/norm(DsS,'fro');
        DdS = DdS/norm(DdS,'fro');
        DsT = DsT/norm(DsT,'fro');
        DdT = DdT/norm(DdT,'fro');
        
        DsST = DsST/norm(DsST,'fro'); 
        DdST = DdST/norm(DdST,'fro'); 
        DsTS = DsTS/norm(DsTS,'fro'); 
        DdTS = DdTS/norm(DdTS,'fro'); 
        
        Ds1=[DsS,DsST];
        Ds2=[DsTS,DsT];
        Dsame=[Ds1; Ds2];
        
        
        Ds1=[DdS,DdST];
        Ds2=[DdTS,DdT];
        Ddiff=[Ds1; Ds2];
        
    else     
        oneM=ones(size(allmeanC,1),size(allmeanC,1));
        D=diag(sum(oneM,1));
        R=2*allmeanC'*D*allmeanC-2*allmeanC'*oneM*allmeanC;
        
         %D same in Sourse
        DsS=zeros(ns,ns);
        for i=1:ns
           nc = length(find(Ys==Ys(i)));
           for j=1:ns
               if i==j
                   DsS(i,j)= ns;
               elseif Ys(i)==Ys(j)
                   DsS(i,j)=-(ns)/nc;
               else
                   DsS(i,j)=0;
               end

            end
        end
        
         
        DdS=zeros(ns,ns);
        for i=1:ns
            nc = length(find(Ys==Ys(i)));
           for j=1:ns
               if i==j
                   DdS(i,j)= ns - nc;
               elseif Ys(i)~= Ys(j)
                   DdS(i,j)= -1;
               else
                   DdS(i,j)= 0;
               end
           end
        end

        DsS = DsS/norm(DsS,'fro');
        DdS = DdS/norm(DdS,'fro');
      
        DsT=zeros(nt,nt);
        DdT=zeros(nt,nt);
        
        Dst=zeros(ns,nt);
        Dts=zeros(nt,ns);
        Ds1=[DsS,Dst];
        Ds2=[Dts,DsT];
        Dsame=[Ds1; Ds2];
        
        
        Ds1=[DdS,Dst];
        Ds2=[Dts,DdT];
        Ddiff=[Ds1; Ds2];
        
    end

	M = M + N;
	M = M / norm(M,'fro');
    
    

	%% Centering matrix H
	H = eye(n) - 1/n * ones(n,n);
    [A,~] = eigs(mu*X*M*X'+allQ+lambda*eye(m)-beta*R + alpha*X*(Dsame-(ru*Ddiff))*X',X*H*X',dim,'SM');
    
    Z = A'*X;
end

