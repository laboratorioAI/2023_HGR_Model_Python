function gesture = AddTestingGestures(handGesture,version)
    
    dofnames = ["x","y","z"];

    for kRep = 1:180
        
        sample = sprintf('idx_%d',kRep);
        
        if  isfield(handGesture{kRep, 1},'groundTruth') && (strcmp('training',version) == 1)
            gesture.(sample).groundTruth = double(handGesture{kRep, 1}.groundTruth);
            gesture.(sample).groundTruthIndex = handGesture{kRep, 1}.groundTruthIndex;        
        end
        
        if (strcmp('testing',version) == 1)
          
        elseif (strcmp('training',version) == 1)
            gesture.(sample).gestureName =  handGesture{kRep, 1}.gestureName;
        end
            
        %gesture.(sample).startPointforGestureExecution = handGesture{kRep, 1}.pointGestureBegins;
        %gesture.(sample).myoDetection = handGesture{kRep, 1}.pose_myo;

        gesture.(sample).quaternion.w = round(handGesture{kRep, 1}.quaternions(:,1), 4); 
        gesture.(sample).quaternion.x = round(handGesture{kRep, 1}.quaternions(:,2), 4);    
        gesture.(sample).quaternion.y = round(handGesture{kRep, 1}.quaternions(:,3), 4); 
        gesture.(sample).quaternion.z = round(handGesture{kRep, 1}.quaternions(:,4), 4);
        for ch = 1:8
          channel = sprintf('ch%d',ch);
          gesture.(sample).emg.(channel) = (handGesture{kRep, 1}.emg(:,ch))*128;
        end

        
       for dof = 1 : 3
           xyz = sprintf('%s',dofnames(dof));
           try
               gesture.(sample).gyroscope.(xyz) = handGesture{kRep, 1}.gyro(:,dof);
               gesture.(sample).accelerometer.(xyz) = round(handGesture{kRep, 1}.accel(:,dof),4);
           catch
               warning('Error: No se tiene esos datos');
           end
       end
        


    end


end