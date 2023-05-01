function synGesture = AddSynchronizationGesture(gesture)

     dofnames = ["x","y","z"];

    for kSyn = 1:5
        
        sample = sprintf('idx_%d',kSyn);
        synGesture.samples.(sample).startPointforGestureExecution  = gesture{kSyn, 1}.pointGestureBegins; 
        synGesture.samples.(sample).deviceDetection                   = gesture{kSyn, 1}.gestureDevicePredicted;
        
                
         synGesture.samples.(sample).quaternion.w = round(gesture{kSyn, 1}.quaternions(:,1), 4); 
         synGesture.samples.(sample).quaternion.x = round(gesture{kSyn, 1}.quaternions(:,2), 4);    
         synGesture.samples.(sample).quaternion.y = round(gesture{kSyn, 1}.quaternions(:,3), 4); 
         synGesture.samples.(sample).quaternion.z = round(gesture{kSyn, 1}.quaternions(:,4), 4);
         
         
         for ch = 1:8               
            channel = sprintf('ch%d',ch);
            synGesture.samples.(sample).emg.(channel) = (gesture{kSyn, 1}.emg(:,ch))*128;       
         end
         
         
         for dof = 1 : 3
            xyz = sprintf('%s',dofnames(dof));
            try
                synGesture.samples.(sample).gyroscope.(xyz) = gesture{kSyn, 1}.gyro(:,dof);
                synGesture.samples.(sample).accelerometer.(xyz) = round(gesture{kSyn, 1}.accel(:,dof),4);
            catch
                warning('Error: No se tiene esos datos');
            end
            
         end
    end
end