function Wb = getWb(hiddenSize, inputSize)
r  = sqrt(8) / sqrt(hiddenSize+inputSize+1);   % range for weights and biases
W1 = rand(hiddenSize, inputSize) * 2 * r - r;
W2 = rand(inputSize, hiddenSize) * 2 * r - r;
b1 = zeros(hiddenSize, 1);
b2 = zeros(inputSize, 1);
Wb = [W1(:) ; W2(:) ; b1(:) ; b2(:)];
end

