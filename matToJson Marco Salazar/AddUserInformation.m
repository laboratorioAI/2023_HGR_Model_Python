function userInformation = AddUserInformation(info)


    userInformation.name   =  info.username;  
    userInformation.age = info.age;
    userInformation.gender = info.gender;
    userInformation.occupation = info.occupation;
    userInformation.ethnicGroup = info.ethnicGroup;
    
    if  info.handedness == "right_Handed"

        userInformation.handedness = 'right';

    elseif info.handedness == "left_Handed"

        userInformation.handedness = 'left';

    end
    
 

    if info.hasSufferedArmDamage == 1

        userInformation.ArmDamage = 'True';

    else

        userInformation.ArmDamage = 'False';

    end

    userInformation.distanceFromElbowToDeviceInCm = info.fromElbowToMyo;
    userInformation.distanceFromElbowToUlnaInCm = info.fromElbowToUlna;
    userInformation.armPerimeterInCm = info.armPerimeter;

    

end