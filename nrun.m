

%Load Detector file , it is Pretrained Neural network for face detection 

img = imread('images\Fd\image_0046.jpg');
% img is Variable , imread is function for reading Image.

[bboxes,scores] = detect(Ndetector,img);
% bboxes = Bounding Boxes which surrounds Face -Rectangle Box
% Scores = Confidence that is how sure a Detector is for identifying Human Face

for i = 1:length(scores)
  
   annotation = sprintf('Confidence = %.1f',scores(i));
   %annotation is labels like face and Confidence in percentage
   
   img = insertObjectAnnotation(img,'rectangle',bboxes(i,:),annotation);
   
end

figure
imshow(img);

img2 = imread('D:\sem5\me\term\images\Fd\image_0190.jpg');
[bboxes,scores] = detect(Ndetector,img2);
% bboxes = Bounding Boxes which surrounds Face -Rectangle Box
% Scores = Confidence that is how sure a Detector is for identifying Human Face

for i = 1:length(scores)
  
   annotation = sprintf('Confidence = %.1f',scores(i));
   %annotation is labels like face and Confidence in percentage
   
   img2 = insertObjectAnnotation(img2,'rectangle',bboxes(i,:),annotation);
   
end

figure
imshow(img2);

img3 = imread('D:\sem5\me\term\images\Fd\image_0048.jpg');
[bboxes,scores] = detect(Ndetector,img3);
% bboxes = Bounding Boxes which surrounds Face -Rectangle Box
% Scores = Confidence that is how sure a Detector is for identifying Human Face

for i = 1:length(scores)
  
   annotation = sprintf('Confidence = %.1f',scores(i));
   %annotation is labels like face and Confidence in percentage
   
   img3 = insertObjectAnnotation(img3,'rectangle',bboxes(i,:),annotation);
   
end

figure
imshow(img3);