function [allQ,R] = Calcu_Q(Xs,Ys,Xt,Yt,label_table)
%Yt are pseudo lables

[Ys_unique,~,~]=unique(Ys,'rows');

allQ=zeros(size(Xs,2),size(Xs,2));
allmeanC=[];
Svalue=min(max(label_table));

for i=1:size(Ys_unique,1)
    index_t= find(Yt==Ys_unique(i,:));
    X_tC=Xt(index_t,:);
    Tabel_T=label_table(index_t,Ys_unique(i,:));
    index_s= find(Ys==Ys_unique(i,:));
    X_sC=Xs(index_s,:);
    
    Xc=[X_sC;X_tC];
    mean_Xs_c=mean(Xc,1);
   
    allmeanC(i,:)=mean_Xs_c;
    
    Xc=[X_sC;X_tC];

    for j=1:size(X_sC,1)
        allQ=(X_sC(j,:)-mean_Xs_c)'*(X_sC(j,:)-mean_Xs_c)/size(Xc,1)*Svalue+allQ;
    end

    for j=1:size(X_tC,1)
        allQ=(X_tC(j,:)-mean_Xs_c)'*(X_tC(j,:)-mean_Xs_c)/size(Xc,1)*Tabel_T(j)+allQ;
    end
    
end
  oneM=ones(size(allmeanC,1),size(allmeanC,1));
  D=diag(sum(oneM,1));
  R=2*allmeanC'*D*allmeanC-2*allmeanC'*oneM*allmeanC;
end

