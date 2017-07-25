function [cost,gradient]= softMaxi(softmaxInput,sm,truth,lambda,numImages,input)
cost = ((-1/numImages).*sum(sum(truth.*log(sm)))) + ((lambda/2).*sum(sum(softmaxInput.^2)));
grad = (-1/numImages).*(((truth-(sm))*input') + (lambda*softmaxInput));
gradient = [grad(:)];
end