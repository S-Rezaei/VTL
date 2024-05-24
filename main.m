% Visual Transductive learning via Iterative Label Correction (VTL)
% Multimedia Systems - Springer - 2024
% The link for the paper : https://link.springer.com/article/10.1007/s00530-024-01339-3


clear;close all;
datapath = 'data/';

options.dim = 30;
options.T = 5;            
options.kernel_type='primal';
options.mu = 0.7;             
options.lambda = 0.5;
options.beta = 0.002; 
options.inIter=10;% 15 or 10 %Sometimes using 10 will have better results.


optionss.alpha = 0.001;            
optionss.ru = 0.01;  



srcStrDecaf = 'amazon';
tgtStrDecaf = 'dslr';


datafeature = 'SURF';

result = [];

Allresult=[];
all_round_result=[];


src = char(srcStrDecaf);
tgt = char(tgtStrDecaf);
options.data = strcat(src,'-vs-',tgt);
fprintf('Data=%s \n',options.data);

% load and preprocess data
data_file = [datapath 'surf/',src '_SURF_10.mat'];
load(data_file);

Xs = feas;              clear feas;
Ys = label;            clear label;

Xs = normr(Xs);

Xs = Xs';

data_file = [datapath 'surf/',tgt '_SURF_10.mat'];
load(data_file);

Xt = feas;              clear feas;
Yt = label;            clear label;
Xt = normr(Xt);

Xt = Xt';

knn_model = fitcknn(Xs',Ys,'NumNeighbors',1);
Cls   = knn_model.predict(Xt');

acc = length(find(Cls==Yt))/length(Yt);
fprintf('original acc = %0.4f\n',full(acc));

fprintf('beta=%0.4f----mu=%0.4f ----lambda=%0.4f\n',full(options.beta),options.mu,options.lambda);
[Z,acc_ite,fixLabel,fixlable_info] =VTL(Xs',Ys,Xt',Yt,options,optionss);

Allresult=cat(3,Allresult,fixlable_info);
all_round_result=cat(3,all_round_result,acc_ite);
maxAcc=max(acc_ite);
result=[result;maxAcc];

result_aver=mean(result);
fprintf('-------------Average accuracy: %0.4f----------------',result_aver*100);

