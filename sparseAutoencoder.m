function [cost,gradient] = sparseAutoencoder(Wb,hiddenSize, inputSize, lambda, beta, rho, patches)
% obtain random values of weights and bias
W1 = reshape(Wb(1:hiddenSize*inputSize),hiddenSize,inputSize);
s=hiddenSize*inputSize;
W2 = reshape(Wb(s+1:s+s),hiddenSize,inputSize);
b1 = Wb(2*s+1:2*s+hiddenSize);
b2 = Wb(2*s+hiddenSize+1:end);
numPatches = size(patches,2); % number of input data taken to train autoencoder - m
% activation
a1 = patches; % activation = input data in first layer
a2 = sigmoid(W1*a1 + repmat(b1,1,numPatches)); % apply activation function - sigmoid
a3 = W2.'*a2 + repmat(b2,1,numPatches); % input data - a2 
% J = 1/m sum(sum(1/2 *(hwb(x) - y).^2)) 
% + lambda/2 sum (Wij(l)) - l = layer , i = input layer, j = hidden layer
J = ((1/numPatches)* sum(sum(1/2 * (a3 - patches).^2))) + ((lambda/2)*(sum(sum(W1.^2)))+ (sum(sum(W2.^2))));
rho_obtained = (1/numPatches)* sum(a2,2);
KL = sum((rho * log(rho./rho_obtained)) + (1-rho) * log((1-rho)./(1-rho_obtained)));
% cost = J + beta*KL
cost = J + (beta*KL);
% Backpropogation
% del value in each layer - the contribution of that unit of data for error
del3 = -(patches-a3); 
del2 = ((W2*del3)+ repmat(beta.*((-rho./rho_obtained)+((1-rho)./(1-rho_obtained))),1,numPatches).*(a2.*(1-a2)));
gradientW2 = ((del3*a2.')/numPatches).' + (lambda*W2);
gradientW1 = ((del2*a1.')/numPatches) + (lambda*W1);
gradientb2 = sum(del3,2)/numPatches;
gradientb1 = sum(del2,2)/numPatches;
% Gradient vector
gradient = [gradientW1(:);gradientW2(:);gradientb1;gradientb2];
end
