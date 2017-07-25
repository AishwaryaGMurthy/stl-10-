function pool_features = CNN(X, hiddenSize)
[row,col]= size(X);
x = reshape(X,3,96,96,row);
Xdim = 96; % dimension of input image
Xchannels =  3; % rgb channels
patchDim = 8 ;  % 8*8 patches to convolve
numPatches = 50000;    % number of patches
inputSize = patchDim * patchDim * Xchannels; % input size to convolve
% For sparsity Encoder - Initialise parameters
%Sparsity Parameter- rho : Desired average activation of hidden layers
rho = 0.01;
% weight decay parameter
lambda = 0.0001;
% weight of sparsity penalty term       
beta = 3;
% get patch
patches = zeros(inputSize, numPatches);
c = 1; % counter
imgIndices =  ceil(rand(1, numPatches) * size(x,4));
pixelRand = ceil(rand(2, numPatches) * (Xdim - patchDim));
pixelRand2 = pixelRand + patchDim - 1;
while c<numPatches
imgIndex = imgIndices(1,c);
p = x(1:3, pixelRand(1,c):pixelRand2(1,c), pixelRand(2,c):pixelRand2(2,c), imgIndex);
patchWork=reshape(p, inputSize,1);
patches(:,c) = patchWork;
c = c+1;
end
% normalize the patches
% remove mean of patch images
patches = bsxfun(@minus, patches, mean(patches));
% Scale to [-1,1]
patches = max(min(patches, (3 * std(patches(:)))), -3 * std(patches(:))) / (3 * std(patches(:)));
% Re-scale - [-1,1] to [0.1,0.9]
patches = (patches + 1) * 0.4 + 0.1;
% get the weights and biases
Wb = getWb(hiddenSize, inputSize);
[cost, gradient] = sparseAutoencoder(Wb,hiddenSize, inputSize, lambda, beta, rho, patches);
% minimize function - use minFuncor fminunc
addpath minFunc/
options.Method = 'lbfgs';
options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';
[optTheta, cost] = minFunc( @(p) sparseAutoencoder(p,hiddenSize, inputSize, lambda, beta, rho, patches),Wb, options);
W = reshape(optTheta(1:inputSize * hiddenSize), hiddenSize, inputSize);
s = hiddenSize*inputSize;
b = optTheta(2*s+1:2*s+hiddenSize);
%perform zca whitening to the obtained weight matrix - to reduce the
%distortion of data while optimizing it
epsilon = 0.1;
[r,c]= size(W);
%regularization matrix
reg = (W'*W + epsilon*eye(c));
r = r + epsilon;
% eigen value of regularization matrix
[r1 c]= eig(reg);
c = max(diag(c),epsilon);
root_inverse=diag(1./sqrt(c));
transformation_matrix=sqrt(r-1)*r1*r1'*root_inverse;
W = W*transformation_matrix;
b = b - W*mean(patches,2);
% initialize the input patch to convolve around the images
convolve_patches = x(:,:,:,1:8);
% call the convolution function to obtain the features to identify through
% convolution - obtain convolved features
conv_features=convolution(convolve_patches,W,b,hiddenSize,patchDim); 
% pooling - the convolved features are pooled together to reduce variance
% Overfitting is avoided
poolDim = 19;          % dimension of pooling region
% obtain the pooled features
pool_features = pooling(conv_features,poolDim); % mean of all convolved features
end