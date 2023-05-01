function generalInformation = AddGeneralInformation(info)

    if  info.DeviceType == "myo"

        generalInformation.deviceModel = 'Myo Armband';
        generalInformation.samplingFrequencyInHertz = 200;

    elseif info.DeviceType == "gForce"

        generalInformation.deviceModel = 'GForce Pro';
        generalInformation.samplingFrequencyInHertz = 500;

    end
    
    generalInformation.recordingTimeInSeconds = 5;
    generalInformation.repetitionsForSynchronizationGesture = 5;
    generalInformation.devicePredictionLabel.noGesture = 0;
    generalInformation.devicePredictionLabel.fist = 1;
    generalInformation.devicePredictionLabel.waveIn = 2;
    generalInformation.devicePredictionLabel.waveOut = 3;   
    generalInformation.devicePredictionLabel.open = 4;
    generalInformation.devicePredictionLabel.pinch = 5;

end