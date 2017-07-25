function pool_features = pooling(conv_features,poolDim)
[numFeatures,numImages,convDim,convDim]=size(conv_features);
% convolved features should be pooled and stored in pool_features
s = floor(convDim / poolDim);
pool_features = zeros(numFeatures,numImages,s,s);
for i=1:numFeatures
    for j=1:numImages
        for start=1:s
            for stop=1:s
              pool_features(i,j,start,stop) = mean(mean(conv_features(i,j,poolDim*(start-1)+1:poolDim*start,poolDim*(stop-1)+1:poolDim*stop)));
            end
        end
    end
end
end