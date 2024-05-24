function [lable_table,lable_table_layers] = UpdataTable(Yt_pre,MaxYs,lable_table,lable_table_layers,iter,Iter)

table_layer=zeros(size(lable_table));
for i=1:size(Yt_pre,1)
    label=Yt_pre(i);
    if label>max(MaxYs)
        continue;
    end
    lable_table(i,label)=lable_table(i,label)+1/Iter;
    table_layer(i,label)=1;
end
lable_table_layers=cat(3,lable_table_layers,table_layer);
end

