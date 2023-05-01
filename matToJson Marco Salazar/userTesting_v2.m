close all
clear all
clc

% Marco SalazaruserData
% Artificial Intelligence and Computer Vision Research Lab
% Escuela Politécnica Nacional, Quito - Ecuador
% marco.salazar02@epn.edu.ec
% 1 Feb, 2023


%% Load training data

folders = dir('testing');
numFolders = length(folders);
version = 'testing';

for i = 001:numFolders
    
    if ~(strcmpi(folders(i).name, '.') || strcmpi(folders(i).name, '..') || strcmpi(folders(i).name, '.DS_Store')) 
        info = load(['testing/' folders(i).name '/userData.mat']) ;
        disp('Processing Data...')
        clc
        userData = info.userData;
        user = sprintf('%s',info.userData.userInfo.username);
        
        % General info of experiment
        userTesting.(user).generalInfo = AddGeneralInformation(userData.deviceInfo);
        
        % User info 
        userTesting.(user).userInfo = AddUserInformation(userData.userInfo); 
        
        % Experiment Date
        userTesting.(user).userInfo.date = datestr(userData.extraInfo.date);
        
        % Synchronization Gesture Samples
        userTesting.(user).synchronizationGesture = AddSynchronizationGesture(userData.sync);
        
        % Training Hand Gestures Samples
        userTesting.(user).trainingSamples = AddTrainingGestures(userData.training);
        
        % Testing Hand Gestures Samples
        userTesting.(user).testingSamples = AddTestingGestures(userData.testing, version);
        
        
    end
    
    
    
    
end

% Conver .mat struct to json dictionary
path = './JSONtesting/';
convertMatToJson(path,userTesting,version)
