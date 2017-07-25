load('test.mat');
hiddenSize = 625;
pool_features = CNN(X, hiddenSize);
numInput = size(X,1);
numlabels = size(class_names,2);
% form softmax layer to convolutional nueral network
s1 = numel(pool_features) / numInput;
input = reshape(pool_features,s1,numInput);
lambda = 0.0001;
options = optimset('MaxFunEvals',8000);
options = struct;
options.Method = 'lbfgs';
options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';
softmaxInput=randn(numlabels*s1,1);
[sOpttheta,cost] = minFunc(@(p)softMax(p,numlabels,s1,lambda,input,y),softmaxInput,options);
sOpttheta=reshape(sOpttheta, numlabels, s1);
%Form the softmax transfer function
sm = softmax(sOpttheta*input);
[t,pred]= max(sm);
% Average accuracy on full test set
acc = sum(pred(:)==y(:))/s1; % accuracy/size 
fprintf('Accuracy = %f\n',acc);