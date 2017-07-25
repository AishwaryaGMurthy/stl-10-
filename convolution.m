function conv_features = convolution(convolve_patches,W,b,numFeatures,patchDim) 
[xChannels,xDim,xDim,numImages]=size(convolve_patches);
features = zeros(xChannels, numFeatures, patchDim*patchDim);
conv_features = zeros(numFeatures,numImages,xDim-patchDim+1,xDim-patchDim+1);
%compute features over every channel
for channel=1:xChannels
  s = (patchDim*patchDim)*(channel-1);
  features(channel,:,:)= W(:,s+1:s+(patchDim*patchDim));
end
for i = 1:numImages
    for j= 1:numFeatures
        % stride = 1, input image = N X N matrix, filter = F X F matrix
        % Convolved Image size = (N - F)+1/stride
        % here convolved image size = (96-8)+1/1=89
        conv_Image = zeros(xDim-patchDim+1,xDim-patchDim+1);       
        for channel = 1:xChannels
            % take each feature separately - form 8X8 filter
            feature = reshape(features(channel,j,:),patchDim,patchDim);
            %flip the feature
            feature = flipud(fliplr(squeeze(feature)));
            % Get the image
            image = squeeze(convolve_patches(channel,:,:,i));
            image = im2double(image);
            % Perform convolution between the input image and the feature
            % Ignore the zero-padded edges while convoluting
            c = conv2(image,feature,'valid');
            conv_Image = conv_Image + c;
        end
        conv_Image = sigmoid(conv_Image + b(j));
        conv_features(j,i,:,:)= conv_Image;
    end
end
end