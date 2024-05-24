function [TheFixLabelYt] = UpdateFix_basedInfo(info_shang,TheFixLabelYt,label_table)


[sortvalue,~]=sort(info_shang);

H=tabulate(sortvalue);

edgevalue=0;
if length(H(:,1))>4
    edgevalue=H(2,1);
else
    if length(H(:,1))>3
        edgevalue=H(2,1);
    else
        if length(H(:,1))>2
            edgevalue=H(1,1);
        else
            if length(H(:,1))>=1
                edgevalue=H(1,1);
            end
        end
    end
end

selectIndex=find(info_shang<edgevalue);
removeIndex=find(info_shang>=edgevalue);

[~,index_clss]=max(label_table(selectIndex,:),[],2);
TheFixLabelYt(selectIndex)=index_clss;
TheFixLabelYt(removeIndex)=0;
end

