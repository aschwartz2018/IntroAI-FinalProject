%% 3.1 Load files from model folder

load('./model/trainedModel.mat');
load('./model/validation.mat');
load('./model/categories.mat');

%% 3.2: Classify the validation images using the fine-tuned network.

[YPred,scores] = classify(modelTransfer,validationSet);

%% 3.3: Calculate the classification accuracy on the validation set. 
% Accuracy is the fraction of labels that the network predicts correctly.

YValidation = validationSet.Labels;
accuracy = mean(YPred == YValidation);
fprintf("The validation accuracy is: %.2f %%\n", accuracy * 100);

%% 3.4: Test it on unseen images
newImage1 = './dog.jpg'; % any dog image should do!
img1 = readAndPreprocessImage(newImage1);
YPred1 = predict(modelTransfer,img1);
[confidence1,idx1] = max(YPred1);
label1 = categories{idx1};
% Display test image and assigned label
figure
imshow(img1)
title(string(label1) + ", " + num2str(100*confidence1) + "%");

newImage2 = './cat.jpg'; % any cat image should do!
img2 = readAndPreprocessImage(newImage2);
YPred2 = predict(modelTransfer,img2);
[confidence2,idx2] = max(YPred2);
label2 = categories{idx2};
% Display test image and assigned label
figure
imshow(img2)
title(string(label2) + ", " + num2str(100*confidence2) + "%");
   
%% 3.5: Test it on unseen images: Your turn!

% What about the iconic "Doge"?
% ENTER YOUR CODE HERE
% ..
% ..
% ..

newImage3 = './doge.jpg'; % any 'doge' image should do!
img3 = readAndPreprocessImage(newImage3);
YPred3 = predict(modelTransfer,img3);
[confidence3, idx3] = max(YPred3);
label3 = categories(idx3);
% Display test image and assigned label
figure
imshow(img3)
title(string(label3) + ", " + num2str(100*confidence3) + "%");