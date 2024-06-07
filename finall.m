clc;
clear all;
close all;

trainImagesDir = fullfile('C:\CODE');
exts = {'.jpg','.bmp','.png'};
pristineImages = imageDatastore(trainImagesDir,'FileExtensions',exts);
upsampledDirName = [trainImagesDir filesep 'upsampledImages'];
residualDirName = [trainImagesDir filesep 'residualImages'];
scaleFactors = [2 3 4];
[filename, pathname]=uigetfile('*.*','pick');
img =imread(filename) ;
img=imresize(img,[312 312]);
figure,imshow(img);
a=fusion(img);
createVDSRTrainingSet(img,pristineImages,scaleFactors,upsampledDirName,residualDirName);
upsampledImages = imageDatastore(upsampledDirName,'FileExtensions','.mat');
residualImages = imageDatastore(residualDirName,'FileExtensions','.mat');
dsTrain1 = [upsampledImages.Files,residualImages.Files];
daTrain=zeros(2);
for i=1:2
for j=1:14
  ds=load(dsTrain1{j,i});
  if j==1 && i==1
  ds=ds.i1;
  elseif j==2 && i==1
  ds=ds.i10;
  elseif j==3 && i==1
  ds=ds.i11;
  elseif j==4 && i==1
  ds=ds.i12;
  elseif j==5 && i==1
  ds=ds.i13;
  elseif j==6 && i==1
  ds=ds.i14;
  elseif j==7 && i==1
  ds=ds.i2;
  elseif j==8 && i==1
  ds=ds.i3;
  elseif j==9 && i==1
  ds=ds.i4;
  elseif j==10 && i==1
  ds=ds.i5;
  elseif j==11 && i==1
  ds=ds.i6;
  elseif j==12 && i==1
  ds=ds.i7;
  elseif j==13 && i==1
  ds=ds.i8;
  elseif j==14 && i==1
  ds=ds.i9;
  elseif j==1 && i==2
  ds=ds.r1;
  elseif j==2 && i==2
  ds=ds.r10;
  elseif j==3 && i==2
  ds=ds.r11;
  elseif j==4 && i==2
  ds=ds.r12;
  elseif j==5 && i==2
  ds=ds.r13;
  elseif j==6 && i==2
  ds=ds.r14;
  elseif j==7 && i==2
  ds=ds.r2;
  elseif j==8 && i==2
  ds=ds.r3;
  elseif j==9 && i==2
  ds=ds.r4;
  elseif j==10 && i==2
  ds=ds.r5;
  elseif j==11 && i==2
  ds=ds.r6;
  elseif j==12 && i==2
  ds=ds.r7;
  elseif j==13 && i==2
  ds=ds.r8;
  elseif j==14 && i==2
  ds=ds.r9;
  end
  dsTrain{j,i}=ds;
end
end
  
 

patchSize = [64 64];

for j=1:14
dsTrain{j,1} = imresize((dsTrain{j,1}),patchSize);

end
for j=1:14
dsTrain{j,2} = imresize((dsTrain{j,2}),patchSize);

end
networkDepth = 20;
firstLayer = imageInputLayer([64 64 1],'Name','InputLayer','Normalization','none');
convLayer = convolution2dLayer(3,11,'Padding',1, ...
    'Name','Conv1');
relLayer = reluLayer('Name','ReLU1');
middleLayers = [convLayer relLayer];
for layerNumber = 2:networkDepth-1
    convLayer = convolution2dLayer(3,64,'Padding',[1 1], ...
         ...
        'Name',['Conv' num2str(layerNumber)]);
    
    relLayer = reluLayer('Name',['ReLU' num2str(layerNumber)]);
    middleLayers = [middleLayers convLayer relLayer];    
end
convLayer = convolution2dLayer(3,1,'Padding',[1 1], ...
     ...
    'NumChannels',64,'Name',['Conv' num2str(networkDepth)]);
finalLayers = [convLayer regressionLayer('Name','FinalRegressionLayer')];
layers = [firstLayer middleLayers finalLayers]';

maxEpochs = 5;
epochIntervals = 1;
initLearningRate = 0.0009;
learningRateFactor = 0.9;
l2reg = 0.00001;
miniBatchSize = 10;
options = trainingOptions('sgdm', ...
    'Momentum',0.9, ...
    'InitialLearnRate',initLearningRate, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',1, ...
    'LearnRateDropFactor',learningRateFactor, ...
    'L2Regularization',l2reg, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThresholdMethod','l2norm', ...
    'GradientThreshold',0.01, ...
    'Verbose',false);

T = cell2table((dsTrain));
net = trainNetwork(T,layers,options);


Ireference = imread(filename);
Ireference = im2double(imresize(Ireference,[312 312]));
figure,
imshow(Ireference)
title('Input Image')
scaleFactor = 0.5;
Ilowres = imresize(Ireference,scaleFactor,'bicubic');
figure,
imshow(Ilowres)
title('Low-Resolution Image')
Iycbcr = rgb2ycbcr(a);
Iy = Iycbcr(:,:,1);
Icb = Iycbcr(:,:,2);
Icr = Iycbcr(:,:,3);
[nrows,ncols,np] = size(a);
Iy_bicubic = imresize(Iy,[nrows ncols],'bicubic');
Icb_bicubic = imresize(Icb,[nrows ncols],'bicubic');
Icr_bicubic = imresize(Icr,[nrows ncols],'bicubic');
Iresidual = activations(net,Iy_bicubic,41);
Iresidual=imresize(Iresidual,[312 312]);
y=Iy_bicubic-uint8(Iresidual);
Ivdsr = ycbcr2rgb(cat(3,double(y),Icb_bicubic,Icr_bicubic));
figure,imshow(mat2gray(Ivdsr));

PSNR=abs(psnr(Ireference,double(Ivdsr)));
k=brisque(Ivdsr);
mae = calMAE(Ireference,double(Ivdsr));
