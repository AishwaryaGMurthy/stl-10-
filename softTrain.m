load('train.mat');
X1=X(fold_indices{1},:);
pool_features = CNN(X1);
numInput = size(X1,1);
numlabels = size(class_names,2);
% form softmax layer to convolutional nueral network
s1 = numel(pool_features) / numInput;
input = reshape(permute(pool_features, [1 3 4 2]),s1,numInput);
lambda = 0.0001;
options = struct;
options.Method = 'lbfgs';
options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';
softmaxInput=randn(numlabels*s1,1);
Y = y(fold_indices{1});
[sOpttheta,cost] = minFunc(@(p)softMax(p,numlabels,s1,lambda,input,Y),softmaxInput,options);
sOpttheta=reshape(sOpttheta, numlabels, s1);
%Form the softmax transfer function
sm = softmax(sOpttheta*input);
[t,pred]= max(sm);
mseTrain = mse(abs(pred-Y));
fprintf('Mean Squared Error in the Training Data: %f\n',mseTrain);