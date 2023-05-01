from Shared import Shared
from glob import glob
import numpy as np
import json
import os
import random
import pandas as pd
import warnings

warnings.filterwarnings("ignore", message="The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.")

pd.options.display.float_format = '{:.20f}'.format
np.set_printoptions(precision=20)

def createDatastore(datastore, labels):
    if not os.path.exists(datastore):
        os.makedirs(datastore)

    # The folders are separated by the device model and
    # One folder is created for each class
    for label in labels:
        path = os.path.join(datastore, 'Myo Armband', label)
        if not os.path.exists(path):
            os.makedirs(path)

    for label in labels:
        path = os.path.join(datastore, 'GForce Pro', label)
        if not os.path.exists(path):
            os.makedirs(path)
    return datastore
    

def generateQFrames(quatSignal, ground_truth, num_gesture_points, gesture_name, emg_sampling_rate):
    quat_sampling_f = 50
    # Allocate space for the results

    num_windows = int(np.floor((quatSignal.shape[1]-(Shared.FRAME_WINDOW_T * quat_sampling_f)) / (Shared.WINDOW_STEP_T * quat_sampling_f)) + 1)
    columnas = ['Quat_Spectrograms', 'Gesture', 'Timestamp']
    #data = [(None, 'noGesture', None, None) for _ in range(num_windows)]
    data = pd.DataFrame(columns=columnas)
    is_included = np.zeros((num_windows), dtype=bool)

    # Creating frames
    for i in range(num_windows):

        # Get signal data to create a frame
        translation = int(np.floor((i)* Shared.WINDOW_STEP_T * quat_sampling_f))
        start = 0 + translation
        end = int((Shared.FRAME_WINDOW_T * quat_sampling_f)+ translation)
        timestamp = start + int(np.floor(Shared.FRAME_WINDOW_T * quat_sampling_f)/ 2)

        if end > len(ground_truth):
            frame_ground_truth = ground_truth[start:]
        else:
            frame_ground_truth = ground_truth[start:end]

        total_ones = np.sum(frame_ground_truth == 1)

        # Get Spectrogram of the window
        frame_signal = quatSignal[:, start:end]
       
        quat_spectrograms = Shared.generate_quat_spectrogram(frame_signal, emg_sampling_rate)

        spectrogramsList = quat_spectrograms.tolist()

        # Set data
        data = data.append({'Quat_Spectrograms': spectrogramsList, 'Gesture': 'noGesture', 'Timestamp': timestamp}, ignore_index=True)


        # Check the threshold to consider gesture
        if total_ones >= Shared.FRAME_WINDOW_T * quat_sampling_f * Shared.TOLERANCE_WINDOW or \
                total_ones >= num_gesture_points * Shared.TOLERNCE_GESTURE_LSTM:
            is_included[i] = True
            data.loc[:,'Gesture'][i] = gesture_name


    # Include no gestures in the sequence
    if Shared.NOGESTURE_FILL == 'all':

        is_included[:] = True

    elif Shared.NOGESTURE_FILL == 'some':

        first = np.where(is_included)[0][0]
        last = np.where(is_included)[0][-1]

        for i in range(1, Shared.NOGESTURE_IN_SEQUENCE + 1):
            # Include some from left
            if first - i >= 0:
                is_included[first - i] = True
            # Include some from right
            if last + i < num_windows:
                is_included[last + i] = True

    data_filtered = data.loc[is_included]
    
    return data_filtered , ground_truth

def generateData(samples, emg_sampling_rate):

    quat_sampling_f = 50

    # Number of noGesture samples to discard them
    no_gesture_per_user = Shared.numGestureRepetitions

    # Allocate space for the results
    transformed_samples = []

    # For each gesture sample
    for i in range(no_gesture_per_user, len(samples)):
        # Get sample data
        sample = samples["idx_"+str(i)]
        gesture_name = sample["gestureName"]
        ground_truth = Shared.processQGroundTruth(sample['groundTruth'],emg_sampling_rate, quat_sampling_f)
        num_gesture_points = int(np.floor((sample['groundTruthIndex'][1] - sample['groundTruthIndex'][0])*(quat_sampling_f/emgSamplingRate)))

        quats = sample["quaternion"]

        # Convertir el diccionario en un array de NumPy
        arregloQuats = np.array([quats[key] for key in quats])

        normalQuat = Shared.normalizeQuaternion(arregloQuats)

        # Generate spectrograms
        data, new_ground_truth = generateQFrames(normalQuat, ground_truth, num_gesture_points,
                                                  gesture_name, emg_sampling_rate)

        # Save the transformed data
        transformed_samples.append((data, gesture_name, new_ground_truth))

    return transformed_samples

def saveSampleInDatastore(samples, user, data_type, data_store, device_type):
    for i, sample in enumerate(samples):
        # Get data from transformed samples
        sequence_data = sample[0]
        class_name = sample[1]
        if class_name == 'relax':
            class_name = 'noGesture'

        # Get data in sequence
        timestamps = sequence_data['Timestamp'].values

        # Create a file name (user-type-sample-start-finish)
        file_name = f"{user.strip()}-{data_type}-{i}-[{int(timestamps[0])}-{int(timestamps[-1])}]"
        
        # Set data to save
        if class_name != 'noGesture':
            newGroundTruth = pd.Series(sample[2])
            dfData = pd.concat([sequence_data, newGroundTruth], axis=1)
        else:
            dfData = sequence_data       

        # Save data
        save_path = os.path.join(data_store, device_type, class_name, file_name + '.json')
        dfData.to_json(save_path)
        

def generateQFramesNoGesture(quatSignal, requestedWindows):

    quat_sampling_f = 50

    # Calculate the number of windows to apply the signal
    numWindows = int(np.floor((quatSignal.shape[1]-(Shared.FRAME_WINDOW_T * quat_sampling_f)) / (Shared.WINDOW_STEP_T * quat_sampling_f)) + 1)
    
    # Calculate if signal needs filling
    numWindowsFill = requestedWindows - numWindows

    if numWindowsFill > 0:
        # Get a nogesture portion of the sample to use as filling
        filling = quatSignal[:,0:int(np.floor(numWindowsFill*Shared.WINDOW_STEP_T * quat_sampling_f))]
        # Fill before frame classification
        quatSignal = np.concatenate((quatSignal, filling), axis=1)
    
    # Allocate space for the results
    columnas = ['Quat_Spectrograms', 'Gesture', 'Timestamp']
    data = pd.DataFrame(columns=columnas)
    
    # For each window
    for i in range(requestedWindows):
        # Get window information
        translation = int(np.floor((i)* Shared.WINDOW_STEP_T * quat_sampling_f))
        inicio = 0 + translation
        finish = int((Shared.FRAME_WINDOW_T * quat_sampling_f)+ translation)
        timestamp = inicio + int(np.floor(Shared.FRAME_WINDOW_T * quat_sampling_f)/ 2)
        
        # Generate a spectrogram
        frameSignal = quatSignal[:,inicio:finish]
        spectrograms = Shared.generate_quat_spectrogram(frameSignal, quat_sampling_f)
        # Get quaternion spectrogram of the window, compare finish value
        # Save data
        data = data.append({'Quat_Spectrograms': spectrograms, 'Gesture': 'noGesture', 'Timestamp': timestamp }, ignore_index=True)

        #NoGesture for all
        data.loc[:,'Gesture'][i] = 'noGesture'

    return data

def generateDataNoGesture(samples, num_frames, emg_sampling_rate):
    # Number of noGesture samples to use them
    no_gesture_per_user = Shared.numGestureRepetitions
    
    # Allocate space for the results
    transformed_samples = []
    
    for i in range(no_gesture_per_user-1):
        # Get sample data
        sample = samples["idx_"+str(i+1)]
        gesture_name = sample["gestureName"]
        quats = sample["quaternion"]

        # Convertir el diccionario en un array de NumPy
        arregloQuats = np.array([quats[key] for key in quats])

        normalQuat = Shared.normalizeQuaternion(arregloQuats)

        if Shared.NOGESTURE_FILL == 'all':
            frames_per_sample = num_frames
        elif Shared.NOGESTURE_FILL == 'some':
            random.seed(i)
            frames_per_sample = random.randint(num_frames[0], num_frames[1])    

        # Generate spectrograms
        data = generateQFramesNoGesture(normalQuat, frames_per_sample)

        # Save the transformed data
        transformed_samples.append((data, gesture_name))
    
    return transformed_samples


# DEFINE THE DIRECTORIES WHERE THE DATA WILL BE FOUND
dataDir = 'DatasetJSON';
trainingDir = 'JSONtraining';

# GET THE USERS DIRECTORIES
users, trainingPath = Shared.get_users(dataDir, trainingDir)

if Shared.includeTesting:
    # Divide in two datasets
    limit = len(users) - Shared.numTestUsers
    usersTrainVal = users[:limit, :]
    usersTest = users[limit:, :]
else:
    usersTrainVal = users

#del dataDir, trainingDir, users, limit

# define the categories
categories = ['fist', 'open', 'pinch', 'waveIn', 'waveOut', 'up', 'down', 'left', 'right', 'forward', 'backward']

# create training and validation datastores
training_datastore = createDatastore('DatastoresLSTMQuat/training', categories)
validation_datastore = createDatastore('DatastoresLSTMQuat/validation', categories)

# create testing datastore if includeTesting is True
if Shared.includeTesting:
    testing_datastore = createDatastore('DatastoresLSTMQuat/testing', categories)

# Clean up variables
#del categories

# GENERATION OF SPECTROGRAMS TO CREATE THE MO#del
if Shared.includeTesting:
    usersSets = [(usersTrainVal, 'usersTrainVal'), (usersTest, 'usersTest')]
else:
    usersSets = [(usersTrainVal, 'usersTrainVal')]
"""

for i in range(len(usersSets)):

    # Select a set of users
    users = usersSets[i][0]
    usersSet = usersSets[i][1]
    
    if usersSet == 'usersTrainVal':
        datastore1, datastore2 = training_datastore, validation_datastore
    elif usersSet == 'usersTest':
        datastore1, datastore2 = testing_datastore, testing_datastore
    
    for user in users:
        # Get user samples
        trainingSamples, validationSamples, emgSamplingRate, deviceType = Shared.get_training_testing_samples(trainingPath, user)
        
        # Transform samples
        transformedSamplesTraining = generateData(trainingSamples, emgSamplingRate)
        transformedSamplesValidation = generateData(validationSamples, emgSamplingRate)

        # Save samples
        saveSampleInDatastore(transformedSamplesTraining, user, 'train', datastore1, deviceType)
        saveSampleInDatastore(transformedSamplesValidation, user, 'validation', datastore2, deviceType)
"""
# Clean up variables
#del validationSamples, transformedSamplesValidation, users, usersTrainVal, usersSet, usersTest


# Include NOGESTURE
# Define the directories where the sequences will be added
if Shared.includeTesting:
    datastores = [training_datastore, validation_datastore, testing_datastore]
else:
    datastores = [training_datastore, validation_datastore]

noGestureFramesPerSample = [None] * len(datastores)

# Clean up variables
#del trainingSamples, transformedSamplesTraining, training_datastore, validation_datastore, testing_datastore


for i in range(len(datastores)):
    
    # Create a file datastore.
    files = glob(datastores[i] + '/**/*.json', recursive=True)

    
    # Check the type of filling
    if Shared.NOGESTURE_FILL == 'all':
        
        # Calulate the mean of frames for all samples
        numFiles = len(files)
        numFramesSamples = np.zeros(numFiles)
        for j in range(numFiles):
            with open(files[j]) as f:
                data = json.load(f)
                frames = data['sequenceData']
                numFramesSamples[j] = len(frames)
        
        # Save the mean of frames for all samples
        noGestureFramesPerSample[i] = round(np.mean(numFramesSamples))
        
    elif Shared.NOGESTURE_FILL == 'some':
        
        # Create labels to identify the class
        labels = Shared.create_labels(files, False)
        gestures = Shared.set_no_gesture_use(False)
        avgNumFramesClass = np.zeros(len(gestures))

        # For each class
        for j in range(len(gestures)):
            
            # Get the files of the class
            gesture = gestures[j]
            idxs = [label == gesture for label in labels]
            filesClass = [files[k] for k in range(len(files)) if idxs[0][k]]
            
            # Get the number of frames of each sample of the class
            numFilesClass = len(filesClass)
            numFramesSamples = np.zeros(numFilesClass)
            for k in range(numFilesClass):
                data = pd.read_json(filesClass[k])
                frames = data['Timestamp'].count()
                numFramesSamples[k] = frames
                 
            # Calculate the mean number of frames for the class
            avgNumFramesClass[j] = round(np.mean(numFramesSamples))
            
        # Save the minimum and maximum number of frames of all classes
        noGestureFramesPerSample[i] = [np.min(avgNumFramesClass), np.max(avgNumFramesClass)]
        
# Clean up variables
#del i, j, k, gesture, gestures, filesClass, files, numFiles, numFilesClass, numFramesSamples, avgNumFramesClass, idxs, labels

# THE STRUCTURE OF THE DATASTORE IS DEFINED
categories = ['noGesture']

trainingDatastore = createDatastore(datastores[0], categories)
validationDatastore = createDatastore(datastores[1], categories)
if Shared.includeTesting:
    testingDatastore = createDatastore(datastores[2], categories)
#del categories, datastores


# GENERATION OF NOGESTURE SPECTROGRAMS TO CREATE THE MO#del

# Get the number of noGesture per dataset
noGestureTraining = noGestureFramesPerSample[0]
noGestureValidation = noGestureFramesPerSample[1]
if Shared.includeTesting:
    noGestureTesting = noGestureFramesPerSample[2]

for i in range(len(usersSets)):
    # Select a set of users
    users = usersSets[i][0]
    usersSet = usersSets[i][1]
    
    if usersSet == 'usersTrainVal':
        noGestureSize1, noGestureSize2, datastore1, datastore2 = noGestureTraining, noGestureValidation, trainingDatastore, validationDatastore
    elif usersSet == 'usersTest':
        noGestureSize1, noGestureSize2, datastore1, datastore2 = noGestureTesting, noGestureTesting, testingDatastore, testingDatastore
  

    for user in users:
        # Get user samples
        trainingSamples, validationSamples, emgSamplingRate, deviceType = Shared.get_training_testing_samples(trainingPath, user)
        # Transform samples
        transformedSamplesTraining = generateDataNoGesture(trainingSamples, noGestureSize1, emgSamplingRate)
        transformedSamplesValidation = generateDataNoGesture(validationSamples, noGestureSize2, emgSamplingRate)
        # Save samples
        saveSampleInDatastore(transformedSamplesTraining, user, 'validation', datastore1, deviceType)
        saveSampleInDatastore(transformedSamplesValidation, user, 'train', datastore2, deviceType)

# Clear variables
#noGestureSize1 = noGestureSize2 = datastore1 = datastore2 = noGestureTesting = noGestureTraining = noGestureValidation = None
#testingDatastore = trainingDatastore = trainingPath = users = usersSet = validationDatastore = transformedSamplesValidation = validationSamples = None

