function convertMatToJson(path,users, type)
    
    userList = fieldnames(users);
    savePath = path; %  String of the path to save your file in
    userNum = size(userList,1);
    ext = '.json';
    
    for i = 1:userNum
        clc  
        fprintf('Processing data from user: %d / %d\n', i, userNum);  
        userchoose = users.(userList{i});
        txt = jsonencode(userchoose);
        newPath = [savePath, userList{i}, '/'];
        %disp(newPath);
        %disp(savePath);
        rutaImagen = [type, '/' , userList{i},  '/photo.png'];
        %disp(rutaImagen);
        %newPath = fullfile(savePath, [fileName '.json']);
        if ~exist(fileparts(newPath), 'dir')
            mkdir(fileparts(newPath)); % crea la carpeta si no existe
        end
        movefile(rutaImagen, newPath); % mueve el archivo a la nueva ruta
        fileID = fopen([newPath userList{i} ext],'wt');
        fprintf(fileID,txt);   
    end


end