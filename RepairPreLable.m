function [Y_tar_pseudo] = RepairPreLable(TheFixLabelYt,Y_tar_pseudo)

differ_index=find(TheFixLabelYt~=Y_tar_pseudo);
removeZeros_index=find(TheFixLabelYt(differ_index,:)>0);
Y_tar_pseudo(removeZeros_index)=TheFixLabelYt(removeZeros_index);

end

