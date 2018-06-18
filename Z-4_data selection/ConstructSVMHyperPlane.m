EventClass = 11;

    c = EventClass;
    load('Class0-11.mat')
    clear ind d;    
    ind = find (label(:,c+1) == 1);
    cnt = 1;
    clear monoInd;
    for i = 1:size(ind,1)
        index = ind(i);        
        if(sum(label(index,:)) == 1)
            monoInd(cnt) = index;
            cnt = cnt + 1;
        end
    end
    %%% Data and Lab belong to the class of interest
    Data = feat(monoInd,:);
    Lab  = c+1;
    r = randperm(size(Data,1));
    train_ind = r(1:floor(size(r,2)*0.8));
    test_ind  = r(floor(size(r,2)*0.8):size(r,2));
    X_train = Data(train_ind,:);
    X_test = Data(test_ind,:);
    %% Data_o and Lab_O belong to the outside class
    temp_o = feat(setdiff(1:length(feat),monoInd),:);    
    clear r;
    rr = randperm(size(temp_o,1));
    r = rr(1:size(Data,1));
    train_ind_o = r(1:floor(size(r,2)*0.8));
    test_ind_o  = r(floor(size(r,2)*0.8):size(r,2));
    X_train_o = temp_o(train_ind_o,:);
    X_test_o  = temp_o(test_ind_o,:);
    X_train = [X_train;X_train_o];
    X_test = [X_test;X_test_o];
    
%Y_train = cell(2*size(X_train,1),1);
clear Y_train;
for i = 1:size(X_train,1)/2
    Y_train(i) = '1';
    Y_train(i+size(X_train,1)/2) = '0';
end
svmStruct  = svmtrain(X_train,Y_train,'autoscale',false,'kktviolationlevel',0.5);  
%svmStructs = fitcsvm(X_train,Y_train');

m = svmStruct;
sv = m.SupportVectors;
alphaHat = m.Alpha;
bias = m.Bias;
kfun = m.KernelFunction;
kfunargs = m.KernelFunctionArgs;
f = kfun(sv,X_test,kfunargs{:})'*alphaHat(:) + bias;
pre = svmclassify(svmStruct,X_test);
acc = size(find(pre(1:size(pre)/2) == '1')) + size(find(pre(size(pre)/2:size(pre)) == '0'));
acc = acc/size(pre);
display(acc);
%% load the generated data, until the accuracy of the SVM stops improving
%% 
epoch = 1800;
name = strcat('GAN_Class_',num2str(c));
name = strcat(name,'/GAN_Class__epoch_',num2str(epoch));
load(name);
data = reshape(arr_0,size(arr_0,1)*size(arr_0,2), size(arr_0,3));
f = kfun(sv,data,kfunargs{:})'*alphaHat(:) + bias;
acc_new = 0;
dist = 0.6;
data_use = [];
step = 399;
%% select the generated samples using the fixed distance
for i =1:step:size(data,1)
    clear X_train_new Y_train_new;
    Y_train_new = Y_train;
    if(i+step > size(data,1))
        break
    end
    f = kfun(sv,data(i:i+step,:),kfunargs{:})'*alphaHat(:) + bias;
    mea = mean(f);
    if(f < dist)
        continue;
    end
    X_train_new= [X_train;data(i:i+step,:)];   
    for s = size(X_train):size(X_train_new)
        Y_train_new(s) = '1';
    end
    svmStruct  = svmtrain(X_train_new,Y_train_new,'autoscale',false,'kktviolationlevel',0.5);  
    pre = svmclassify(svmStruct,X_test);
    acc_new = size(find(pre(1:size(pre)/2) == '1')) + size(find(pre(size(pre)/2:size(pre)) == '0'));
    acc_new = acc_new/size(pre);    
    if(acc_new < acc)
        continue;
    end
    data_use = [data_use;data(i:i+step,:)];
    acc = acc_new    
end
%% select the generated samples using the moving dist
% while (acc_new < acc)
%     ind = find(f > dist);
%     clear new_data;
%     new_data = data(ind(1:500),:); 
%     Y_train_new = Y_train;
%     X_train_new= [X_train;new_data];   
%     for i = size(X_train):size(X_train_new)
%         Y_train_new(i) = '1';
%     end    
%     
%     svmStruct  = svmtrain(X_train_new,Y_train_new,'autoscale',false);
%     pre = svmclassify(svmStruct,X_test);
%     acc_new = size(find(pre(1:size(pre)/2) == '1')) + size(find(pre(size(pre)/2:size(pre)) == '0'));
%     acc_new = acc_new/size(pre);
%     
%     if(acc_new < acc) 
%         dist = dist + 0.01;
%         acc_new
%         dist
%         continue;
%     end  
%     %% save the generated data after filtering  data_new
%     
%   
% end  
arr_1 = zeros(size(data_use,1),11);
arr_1(:,EventClass+1) = 1;
arr_0 = data_use;
name = strcat('GAN_After_SVM/GAN_AfterSVM_Class_',num2str(EventClass));
save('GAN_After_SVM/GAN_Class_glassjingling','arr_0','arr_1');
save(name,'arr_0','arr_1');

 