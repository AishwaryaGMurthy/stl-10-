function [cost,gradient]= softMax(softmaxInput,numlabels,inputSize,lambda,input,y)
softmaxInput = reshape(softmaxInput,numlabels,inputSize);
numImages = size(inputSize,2);
truth=full(sparse(y, 1:numImages, 1));
%Form the softmax transfer function
sm = softmax(softmaxInput*input);
%Cost
cost = ((-1/numImages).*sum(sum(truth.*log(sm)))) + ((lambda/2).*sum(sum(softmaxInput.^2)));
grad = (-1/numImages).*(((truth-(sm))*input') + (lambda*softmaxInput));
gradient = [grad(:)];
end