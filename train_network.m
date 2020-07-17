%% 1.1: Loading a pre-trained "AlexNet"

model = alexnet;

%% 2.1 Load larger dataset and build image store

dataFolder = './data/';
categories = {'cat*', 'dog*'};

save('./model/categories.mat','categories');

imds = imageDatastore(fullfile(dataFolder, categories), 'LabelSource', 'foldernames');

% Make the labels of imds correct
myFiles = imds.Files;
for i=1:size(myFiles)
    [filepath,name,ext] = fileparts(string(myFiles(i)));
    myNames(i) = extractBefore(name,'.');
end
imds.Labels = categorical(myNames);

tbl = countEachLabel(imds);
disp (tbl)

% Use the smallest overlap set
% (useful when the two classes have different number of elements but not
% needed in this case)
minSetCount = min(tbl{:,2});

% Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');

% Notice that each set now has exactly the same number of images.
countEachLabel(imds)

%% 2.2: Pre-process Images For CNN
% AlexNet can only process RGB images that are 227-by-227.
% To avoid re-saving all the images to this format, setup the |imds|
% read function, |imds.ReadFcn|, to pre-process images on-the-fly.
% The |imds.ReadFcn| is called every time an image is read from the
% |ImageDatastore|.
%
% Set the ImageDatastore ReadFcn
imds.ReadFcn = @(filename)readAndPreprocessImage(filename);

%% 2.3: Divide data into training and validation sets, then save validation set
[trainingSet, validationSet] = splitEachLabel(imds, 0.8, 'randomized');

save('./model/validation.mat','validationSet');

countEachLabel(trainingSet)
countEachLabel(validationSet)

%% 2.4: Freeze all but last three layers

layersTransfer = model.Layers(1:end-3);
numClasses = 2; % cat and dog

layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

%% 2.5: Configure training options

options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',validationSet, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

%% 2.6: Retrain network and save to file
 
modelTransfer = trainNetwork(trainingSet,layers,options);
save('./model/trainedModel.mat','modelTransfer');

%% References
% [1] Deng, Jia, et al. "Imagenet: A large-scale hierarchical image
% database." Computer Vision and Pattern Recognition, 2009. CVPR 2009. IEEE
% Conference on. IEEE, 2009.
%
% [2] Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet
% classification with deep convolutional neural networks." Advances in
% neural information processing systems. 2012.
%
% [3] Vedaldi, Andrea, and Karel Lenc. "MatConvNet-convolutional neural
% networks for MATLAB." arXiv preprint arXiv:1412.4564 (2014).
%
% [4] Zeiler, Matthew D., and Rob Fergus. "Visualizing and understanding
% convolutional networks." Computer Vision-ECCV 2014. Springer
% International Publishing, 2014. 818-833.
%
% [5] Donahue, Jeff, et al. "Decaf: A deep convolutional activation feature
% for generic visual recognition." arXiv preprint arXiv:1310.1531 (2013).
