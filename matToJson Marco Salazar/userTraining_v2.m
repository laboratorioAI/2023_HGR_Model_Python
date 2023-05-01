close all
clear all
clc

% Marco Salazar
% Artificial Intelligence and Computer Vision Research Lab
% Escuela Politécnica Nacional, Quito - Ecuador
% marco.salazar02@epn.edu.ec
% 1 Feb, 2023


%% Load training data

folders = dir('training');
numFolders = length(folders);
version = 'training';

for i = 001:numFolders
    
    if ~(strcmpi(folders(i).name, '.') || strcmpi(folders(i).name, '..') || strcmpi(folders(i).name, '.DS_Store')) 
        info = load(['training/' folders(i).name '/userData.mat']) ;
        disp('Processing Data...')
        clc
        userData = info.userData;
        user = sprintf('%s',info.userData.userInfo.username);
        
        % General info of experiment
        userTraining.(user).generalInfo = AddGeneralInformation(userData.deviceInfo);
        
        % User info 
        userTraining.(user).userInfo = AddUserInformation(userData.userInfo); 
        
        % Experiment Date
        userTraining.(user).userInfo.date = datestr(userData.extraInfo.date);
        
        % Synchronization Gesture Samples
        userTraining.(user).synchronizationGesture = AddSynchronizationGesture(userData.sync);
        
        % Training Hand Gestures Samples
        userTraining.(user).trainingSamples = AddTrainingGestures(userData.training);
        
        % Testing Hand Gestures Samples
        userTraining.(user).testingSamples = AddTestingGestures(userData.testing, version);
              
    end
    
    
    
    
end

% Conver .mat struct to json dictionary
path = './JSONtraining/';
convertMatToJson(path,userTraining, version)

