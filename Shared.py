import os
import random
import json
import numpy as np
import scipy.signal as signal
from scipy.signal import filtfilt, butter, get_window
from pathlib import Path

class Shared:
    # Spectrogram
    FRECUENCIES = list(range(13))
    WINDOW = 24
    OVERLAPPING = int(WINDOW * 0.5)
    
    # Frame
    FRAME_WINDOW = 300
    WINDOW_STEP = 15 # To obtain the frames
    # Quat frame
    Q_FRAME_WINDOW = 75
    Q_WINDOW_STEP = 4
    # if frame > TOLERANCE_WINDOW || frame > TOLERNCE_GESTURE -> gesture
    TOLERANCE_WINDOW = 0.75
    TOLERNCE_GESTURE = 0.5 # 0.75 0.25; 

    # Time constants
    WINDOW_T = 0.12
    OVERLAPPING_T = WINDOW_T * 0.5
    FRAME_WINDOW_T = 1.5
    WINDOW_STEP_T = 0.075
    
    # Recognition
    WINDOW_STEP_RECOG = 15 # 15 30
    FRAME_CLASS_THRESHOLD = 0.5 # 0.75 0.25;
    # if labels > MIN_LABELS_SEQUENCE -> true
    MIN_LABELS_SEQUENCE = 4
    FILLING_TYPE = 'before' # 'before' 'none'
    POSTPROCESS = '1-1' # '1-1' '1-2' '2-1'
    
    # Evaluation
    FILLING_TYPE_EVAL = 'none' # 'before' 'none'
    
    # For LSTM
    FILLING_TYPE_LSTM = 'before' # 'before' 'none'
    NOGESTURE_FILL = 'some' # 'some' 'all'
    NOGESTURE_IN_SEQUENCE = 6 # if 'some'
    WINDOW_STEP_LSTM = 15 # 15 30
    PAD_KIND = 'shortest' # 'shortest' 'longest'
    TOLERNCE_GESTURE_LSTM = 0.5 # 0.75 0.25;
    NUM_HIDDEN_UNITS = 128 # 128 %58(60) %27(30)
    
    # Samples and signals
    numSamplesUser = 180
    numGestureRepetitions = 16
    numChannels = 8
    
    # User distribution
    includeTesting = False #False True
    numTestUsers = 16

    @staticmethod
    def get_users(data_dir, sub_dir):
        data_path = os.path.join(data_dir, sub_dir)
        users = os.listdir(data_path)
        users = [user for user in users if os.path.isdir(os.path.join(data_path, user))]
        random.seed(9)
        random.shuffle(users)
        return users, data_path

    @staticmethod
    def get_training_testing_samples(data_dir, user):
        file_path = os.path.join(data_dir, user, user + '.json')
        with open(file_path) as f:
            user_data = json.load(f)
        #print(user)
        # Extract samples
        emg_sampling_rate = user_data['generalInfo']['samplingFrequencyInHertz']
        user_name = user_data['userInfo']['name']
        training_samples = user_data['trainingSamples']
        testing_samples = user_data['testingSamples']
        device_type = user_data['generalInfo']['deviceModel']
        return training_samples, testing_samples, emg_sampling_rate, device_type, user_name

    @staticmethod
    def read_file(filename):
        with open(filename, 'r') as file:
            data = json.load(file)
        return data
    
    @staticmethod
    def get_signal(emg):
        # Get channels
        channels = emg.keys()
        # Signal dimensions
        signal = np.zeros((len(emg[channels[0]]), len(channels)))
        for i, channel in enumerate(channels):
            signal[:, i] = emg[channel]
        return signal
    
    @staticmethod
    def rectify_emg(raw_emg, rect_fcn):
        if rect_fcn == 'square':
            rectified_emg = np.square(raw_emg)
        elif rect_fcn == 'abs':
            rectified_emg = np.abs(raw_emg)
        elif rect_fcn == 'none':
            rectified_emg = raw_emg
        else:
            print('Wrong rectification function. Valid options are square, abs and none')
            rectified_emg = None
        return rectified_emg
    
    @staticmethod
    def pre_process_emg_segment(EMGsegment_in, Fa, Fb, rectFcn):
        # Normalization
        # Convertir el diccionario en un array de NumPy
        arrayEMGsegment_in = np.array([EMGsegment_in[key] for key in EMGsegment_in])
        if np.max(np.abs(arrayEMGsegment_in)) > 1:
            EMGnormalized = arrayEMGsegment_in / 128
            #print("Se normaliza")
        else:
            EMGnormalized = arrayEMGsegment_in

        EMGrectified = Shared.rectify_emg(EMGnormalized, rectFcn)

        EMGsegment_out = filtfilt(Fb, Fa, EMGrectified)

        return EMGsegment_out

    @staticmethod
    def preprocess_signal(signal):
        # Butterworth filter coefficients for low-pass filter at 10 Hz
        order = 5  # filter order
        fc_normalized = 0.1
        Fb, Fa = butter(order, fc_normalized, btype='low')

        signal_filt = Shared.pre_process_emg_segment(signal, Fa, Fb, 'abs')

        return signal_filt

    @staticmethod
    def generate_spectrograms(signalIn, sampleFrequency):
        numCols = np.floor((signalIn.shape[1] - Shared.OVERLAPPING) / (Shared.WINDOW - Shared.OVERLAPPING))
        spectrograms = np.zeros((len(Shared.FRECUENCIES), int(numCols), Shared.numChannels))
        for i in range(signalIn.shape[0]):
            f, t, sxx = signal.spectrogram(signalIn[i, :], fs=sampleFrequency, detrend=False, nfft=200, nperseg=24, window=get_window('hamming',24), noverlap=12, mode='complex', scaling='density', axis=0)
            spectrogram = abs(sxx[:13])**2
            spectrograms[:, :, i] = spectrogram#normalized_spectrogram
        return spectrograms

    @staticmethod
    def generate_quat_spectrogram(quatSignal, samplingRate):
        numCols = np.floor((quatSignal.shape[0] - Shared.OVERLAPPING) / (Shared.WINDOW - Shared.OVERLAPPING))
        frecuenciaMuestreo = samplingRate
        quatSpectrograms = np.zeros((len(Shared.FRECUENCIES), 24, 4))
        #print(numCols)
        for i in range(quatSignal.shape[1]):
            f, t, m = signal.spectrogram(quatSignal[:, i], fs=frecuenciaMuestreo, detrend=False, nfft=200, nperseg=6, window=get_window('hamming', 6), noverlap=3, mode='complex', scaling='density', axis=0)
            quatSpectrogram = abs(m[:13])**2
            quatSpectrograms[:, :, i] = quatSpectrogram
        return quatSpectrograms
        
    @staticmethod
    def normalizeQuaternion(quatMatrix):
        normalizedQuat = np.zeros((quatMatrix.shape[0], quatMatrix.shape[1]))
        for j in range(quatMatrix.shape[1]):
            quatChannel = quatMatrix[:, j]
            minValue = np.min(quatChannel)
            maxValue = np.max(quatChannel)
            normalizedQuat[:, j] = (quatChannel - minValue) / (maxValue - minValue)
        return normalizedQuat
    
    @staticmethod
    def expand_signal(quatChannel, emgSamplingRate):
        ratio = emgSamplingRate / 50
        n = np.arange(1, ratio + 1)
        p = quatChannel
        sizeP = p.shape[0]
        p = np.concatenate((p, [p[sizeP - 1]]))
        pnew = []
        for i in range(sizeP):
            a = p[i]
            b = p[i + 1]
            dif = (a - b) / ratio
            element = p[i] - dif * (n - 1)
            pnew.extend(element)
        return np.array(pnew)

    
    @staticmethod
    def set_no_gesture_use(withNoGesture):
        if withNoGesture:
            classes = ["fist", "noGesture", "open", "pinch", "waveIn", "waveOut", "up", "down", "left", "right", "forward", "backward"]
        else:
            classes = ["fist", "open", "pinch", "waveIn", "waveOut", "up", "down", "left", "right", "forward", "backward"]
        return classes
    

    @staticmethod
    def create_labels(files, withNoGesture):
        # Get the number of files
        numObservations = len(files)
        
        # Allocate space for labels
        labels = [None]*numObservations
        
        for i in range(numObservations):
            file = files[i]
            filepath = Path(file).parent  # ../datastore/class
            # The last part of the path is the label
            _, label = filepath.parts[-2:]  # [../datastore, class]
            labels[i] = label
        
        classes = Shared.set_no_gesture_use(withNoGesture)
        labels = np.array(labels)
        labels = np.array([c if c in classes else 'noGesture' for c in labels])
        
        return labels, numObservations
    
    @staticmethod
    def processQGroundTruth(groundTArray, emgSamplingF, quatSamplingF):
        POINT_INTERVAL = emgSamplingF / quatSamplingF
        TRESHOLD = 0.75
        gtArrayLen = len(groundTArray)
        indexes = np.arange(0, gtArrayLen, POINT_INTERVAL, dtype=int)
        quatGroudTruth = np.zeros(len(indexes), dtype=int)
        for z in range(1, len(indexes)):
            sumInterval = np.sum(groundTArray[indexes[z-1]:indexes[z]-1])
            if sumInterval >= np.floor(POINT_INTERVAL * TRESHOLD):
                quatGroudTruth[z-1] = 1
        if indexes[-1] != gtArrayLen:
            sumInterval = np.sum(groundTArray[indexes[-1]:gtArrayLen])
            if sumInterval >= np.floor((gtArrayLen - indexes[-1]) * TRESHOLD):
                quatGroudTruth[-1] = 1
        return quatGroudTruth