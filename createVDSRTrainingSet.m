function createVDSRTrainingSet(filename,pristineImages,scaleFactors,upsampledDirName,residualDirName);
% n=numel(pristineImages.Files);
% for i=1:1
% a=imread(pristineImages.Files{i});
% fusion;

img=imresize(filename,[312 312]); 
% figure,imshow(img);
a=fusion(img);
a1=rgb2ycbcr(a);
ycom=a1(:,:,1);
scaleFactor = 0.25;
Ilowres = imresize(ycom,scaleFactor,'bicubic');
[nrows,ncols,~] = size(ycom);
Iy_bicubic = imresize(Ilowres,[nrows ncols],'bicubic');
res=double(Iy_bicubic)-double(ycom);
figure,imshow(mat2gray(res));
% save res
% save Iy_bicubic

