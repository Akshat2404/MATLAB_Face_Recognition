% Face Detection Steps :-
% 1.Image labelling
% 2.Training
% 3.Testing

load('nosed.mat');  
%  1. Load labelled data file ,which created through image labelling

Facedetect = selectLabels(gTruth,'Nose');
% 2. Creating Variable facedetect in which will store labels 'lips'

if isfolder(fullfile('NoseTrainingData'))    
cd NoseTrainingData
else 
mkdir NoseTrainingData
end
addpath('NoseTrainingData');


% 3. if else condition, if means 'if full name NoseTrainingData exist ,locate that file ' else here if NoseTrainingData Files doesnt exits Make one and 
%         it to the MATLAB Path

trainingData = objectDetectorTrainingData(Facedetect,'SamplingFactor',1,'writeLocation','NoseTrainingData');

% Make variable trainingData in which will store and passing Parameters like Facedetect that is labels , Sampling factor means Examples Face images
% if sampling factor is 2 than 2times negative images taken, Writing location = as TrainingData Folder

Ndetector = trainACFObjectDetector(trainingData,'NumStages',20);

% detector is variable storing data of ACF Object Detector Neural network , Numstages= Number of Training stages,
% More stages like 10,20 takes long time to train but with higher Accuracy.
% Also For No. of Stages also Depends on Number of Positive Smaple image i.e the image we have labelled

save('NDetector.mat','Ndetector');

% saving Detector file , so once ACF detector trained ,it can be used to detect Faces 

rmpath('NoseTrainingData');

%Saving detector file in TrainingData Folder  
%Upto this 13 lines of Code , It needs to run Only Once .
%once we have save Our Neural Network 'Detector.mat' file which detects faces . one Have saved in TrainingData folder ,
% So to use it whenever we just need to load it by specfying  its path 

% Once detector is Trained. 
% Above codes , Not needed to Run again and again
% 
% Below Codes are to be Run.

load('NDetector.mat');

%Load Detector file , it is Pretrained Neural network for face detection 



