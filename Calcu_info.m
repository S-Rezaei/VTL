function [info_shang] = Calcu_info(Table)

probla_Table=inv(diag(sum(Table,2)))*Table;
info_shang=sum(-1*probla_Table.*log2(probla_Table),2);
end

