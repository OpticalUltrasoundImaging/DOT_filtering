clear all
close all
load('diagnostic_testresult.mat');
load('patient_hemo.mat')
hemo_all = [];
indexes = ones(1,size(inputs_all,1));
benign_indexes = ones(1,size(inputs_all,1));
mali_indexes = ones(1,size(inputs_all,1));
% inputs_contrast = [];
% outputs_contrast = [];
% third_inputs_contrast = [];
% third_outputs_contrast = [];
inputs_hemo_contrast = zeros(1,length(inputs_all)/4);
outputs_hemo_contrast = zeros(1,length(inputs_all)/4);
third_inputs_hemo_contrast = zeros(1,length(inputs_all)/4);
third_outputs_hemo_contrast = zeros(1,length(inputs_all)/4);
for idx = 1:size(inputs_all,1)/4
    W_740_780_808_830=2.303*[1.3029 0.4383; 1.1050 0.7360; 0.8040 0.9164; 0.7804 1.0507];
    W_square=W_740_780_808_830'*W_740_780_808_830;
    KK=W_square\W_740_780_808_830';
    K=sum(KK);
    
    volume74 = squeeze(inputs_all((idx-1)*4+1,1:7,:,:));
    volume78 = squeeze(inputs_all((idx-1)*4+2,1:7,:,:));
    volume80 = squeeze(inputs_all((idx-1)*4+3,1:7,:,:));
    volume83 = squeeze(inputs_all((idx-1)*4+4,1:7,:,:));
    start_layer = patient_hemo(9+idx,6);
    layers = patient_hemo(9+idx,5);
    
    numOfSubplots = size(volume74, 1);
    for p=1:numOfSubplots
        for i=1:64
            for j=1:64
                total_oxy(p,j,i)=K(1)*volume74(p,j,i)+K(2)*volume78(p,j,i)+K(3)*volume80(p,j,i)+K(4)*volume83(p,j,i);
                deoxy(p,j,i)=KK(1,1)*volume74(p,j,i)+KK(1,2)*volume78(p,j,i)+KK(1,3)*volume80(p,j,i)+KK(1,4)*volume83(p,j,i);
                oxy(p,j,i)=KK(2,1)*volume74(p,j,i)+KK(2,2)*volume78(p,j,i)+KK(2,3)*volume80(p,j,i)+KK(2,4)*volume83(p,j,i);
            end
        end
    end
    hemo=(total_oxy)*1000;
    deoxy=deoxy*1000;
    oxy=oxy*1000;
    if layers >= 2
        input74 = max(squeeze(volume74(start_layer+1,:,:)),[],'all')/max(squeeze(volume74(start_layer,:,:)),[],'all');
        input78 = max(squeeze(volume78(start_layer+1,:,:)),[],'all')/max(squeeze(volume78(start_layer,:,:)),[],'all');
        input80 = max(squeeze(volume80(start_layer+1,:,:)),[],'all')/max(squeeze(volume80(start_layer,:,:)),[],'all');
        input83 = max(squeeze(volume83(start_layer+1,:,:)),[],'all')/max(squeeze(volume83(start_layer,:,:)),[],'all');
        inputs_contrast((idx-1)*4+1:(idx-1)*4+4) = [input74,input78,input80,input83];
        inputs_hemo_contrast(idx) = max(squeeze(hemo(start_layer+1,:,:)),[],'all')/max(squeeze(hemo(start_layer,:,:)),[],'all');
        if layers >= 3
            input74 = max(squeeze(volume74(start_layer+2,:,:)),[],'all')/max(squeeze(volume74(start_layer,:,:)),[],'all');
            input78 = max(squeeze(volume78(start_layer+2,:,:)),[],'all')/max(squeeze(volume78(start_layer,:,:)),[],'all');
            input80 = max(squeeze(volume80(start_layer+2,:,:)),[],'all')/max(squeeze(volume80(start_layer,:,:)),[],'all');
            input83 = max(squeeze(volume83(start_layer+2,:,:)),[],'all')/max(squeeze(volume83(start_layer,:,:)),[],'all');
            third_inputs_contrast((idx-1)*4+1:(idx-1)*4+4) = [input74,input78,input80,input83];
            third_inputs_hemo_contrast(idx) = max(squeeze(hemo(start_layer+2,:,:)),[],'all')/max(squeeze(hemo(start_layer,:,:)),[],'all');
        end
    else
        
    end
    figure(5)
    for p=1:numOfSubplots
        subplot(3,3,p);

        h = imagesc(squeeze(hemo(p,:,17:49))',[0 100]);
        set(gca,'YDir','normal')
        colormap('jet');
        colorbar
    end
    set(gcf, 'Position',  [100, 100, 600, 400])
    max_input_hemo = max(hemo,[],'all');
    saveas(gcf,[num2str(patient_hemo(idx + 9,1)) 'input_hemo.png'])
    
    volume74 = squeeze(outputs_all((idx-1)*4+1,1:7,:,:));
    volume78 = squeeze(outputs_all((idx-1)*4+2,1:7,:,:));
    volume80 = squeeze(outputs_all((idx-1)*4+3,1:7,:,:));
    volume83 = squeeze(outputs_all((idx-1)*4+4,1:7,:,:));
    
    
    numOfSubplots = size(volume74, 1);
    for p=1:numOfSubplots
        for i=1:64
            for j=1:64
                total_oxy(p,j,i)=K(1)*volume74(p,j,i)+K(2)*volume78(p,j,i)+K(3)*volume80(p,j,i)+K(4)*volume83(p,j,i);
                deoxy(p,j,i)=KK(1,1)*volume74(p,j,i)+KK(1,2)*volume78(p,j,i)+KK(1,3)*volume80(p,j,i)+KK(1,4)*volume83(p,j,i);
                oxy(p,j,i)=KK(2,1)*volume74(p,j,i)+KK(2,2)*volume78(p,j,i)+KK(2,3)*volume80(p,j,i)+KK(2,4)*volume83(p,j,i);
            end
        end
    end
    hemo=(total_oxy)*1000;
    deoxy=deoxy*1000;
    oxy=oxy*1000;
    
    if layers >= 2
        input74 = max(squeeze(volume74(start_layer+1,:,:)),[],'all')/max(squeeze(volume74(start_layer,:,:)),[],'all');
        input78 = max(squeeze(volume78(start_layer+1,:,:)),[],'all')/max(squeeze(volume78(start_layer,:,:)),[],'all');
        input80 = max(squeeze(volume80(start_layer+1,:,:)),[],'all')/max(squeeze(volume80(start_layer,:,:)),[],'all');
        input83 = max(squeeze(volume83(start_layer+1,:,:)),[],'all')/max(squeeze(volume83(start_layer,:,:)),[],'all');
        outputs_contrast((idx-1)*4+1:(idx-1)*4+4) = [input74,input78,input80,input83];
        outputs_hemo_contrast(idx) =  max(squeeze(hemo(start_layer+1,:,:)),[],'all')/max(squeeze(hemo(start_layer,:,:)),[],'all');
        if layers >= 3
            input74 = max(squeeze(volume74(start_layer+2,:,:)),[],'all')/max(squeeze(volume74(start_layer,:,:)),[],'all');
            input78 = max(squeeze(volume78(start_layer+2,:,:)),[],'all')/max(squeeze(volume78(start_layer,:,:)),[],'all');
            input80 = max(squeeze(volume80(start_layer+2,:,:)),[],'all')/max(squeeze(volume80(start_layer,:,:)),[],'all');
            input83 = max(squeeze(volume83(start_layer+2,:,:)),[],'all')/max(squeeze(volume83(start_layer,:,:)),[],'all');
            third_outputs_contrast((idx-1)*4+1:(idx-1)*4+4) = [input74,input78,input80,input83];
            third_outputs_hemo_contrast(idx) =  max(squeeze(hemo(start_layer+2,:,:)),[],'all')/max(squeeze(hemo(start_layer,:,:)),[],'all');
        end
    end
    
    figure(10)
    for p=1:numOfSubplots
        subplot(3,3,p);

        h = imagesc(squeeze(hemo(p,:,17:49))',[0 100]);
        set(gca,'YDir','normal')
        colormap('jet');
        colorbar
    end
    set(gcf, 'Position',  [100, 100, 600, 400])
    max_outputs_hemo = max(hemo,[],'all');
    saveas(gcf,[num2str(patient_hemo(idx + 9,1)) 'output_hemo.png'])
    hemo_all = [hemo_all;[max_input_hemo,max_outputs_hemo]];
    if idx == 51
        a=1;
    end
    if max_input_hemo < 45
        indexes((idx-1)*4+1:(idx-1)*4+4) = 0;
    end
    if max_input_hemo < 45 || patient_hemo(9+idx,4) == 1
        benign_indexes((idx-1)*4+1:(idx-1)*4+4) = 0;
    end
    if max_input_hemo < 45 || patient_hemo(9+idx,4) == 0
        mali_indexes((idx-1)*4+1:(idx-1)*4+4) = 0;
    end
end
layer_idx = zeros(size(indexes));
bormali = zeros(size(indexes));
for i = 1:size(indexes,2)/4
    layer_idx((i-1)*4+1:(i-1)*4+4) = patient_hemo(9+i,5);
    bormali((i-1)*4+1:(i-1)*4+4) = patient_hemo(9+i,4);
end
second_idx = (layer_idx >= 2);
third_idx = (layer_idx >= 3);
inputs_contrast1 = inputs_contrast(and(second_idx,logical(indexes)));
outputs_contrast1 = outputs_contrast(and(second_idx,logical(indexes)));
third_inputs_contrast1 = third_inputs_contrast(and(third_idx,logical(indexes)));
third_outputs_contrast1 = third_outputs_contrast(and(third_idx,logical(indexes)));
second = [inputs_contrast1',outputs_contrast1'];
third = [third_inputs_contrast1',third_outputs_contrast1'];
figure;boxplot(third,{'Reconstruction','NN output'})
title('Second layer contrast')

indexes_hemo = indexes(1:4:end);
diff_hemo = hemo_all(logical(indexes_hemo),2)-hemo_all(logical(indexes_hemo),1);
diff_hemo = diff_hemo./hemo_all(logical(indexes_hemo),1);

inputs_hemo_contrast1 = inputs_hemo_contrast(logical(indexes_hemo));
outputs_hemo_contrast1 = outputs_hemo_contrast(logical(indexes_hemo));
third_inputs_hemo_contrast1 = third_inputs_hemo_contrast(logical(indexes_hemo));
third_outputs_hemo_contrast1 = third_outputs_hemo_contrast(logical(indexes_hemo));
second_hemo = [nonzeros(inputs_hemo_contrast1),nonzeros(outputs_hemo_contrast1)];
third_hemo = [nonzeros(third_inputs_hemo_contrast1),nonzeros(third_outputs_hemo_contrast1)];
patient_hemo(10:end,2:3) = hemo_all;

input_max = max(inputs_all(:,1:7,:,:),[],[2,3,4]);
output_max = max(outputs_all(:,1:7,:,:),[],[2,3,4]);
benigns = [input_max(bormali == 0),output_max(bormali == 0)];
malis = [input_max(bormali == 1),output_max(bormali == 1)];
figure;boxplot(second_hemo,{'Reconstruction','NN output'})
title('Second layer contrast')
data = [input_max(logical(benign_indexes))',output_max(logical(benign_indexes))',input_max(logical(mali_indexes))',output_max(logical(mali_indexes))'];
labels = [zeros(1,length(input_max(logical(benign_indexes)))),1+zeros(1,length(output_max(logical(benign_indexes)))),2+zeros(1,length(input_max(logical(mali_indexes)))),3+zeros(1,length(output_max(logical(mali_indexes))))];
figure;boxplot(data,labels)

hemo_all = hemo_all(2:end,:);
hemo_bormali = patient_hemo(11:end,4);
good_hemo = hemo_all(:,1) > 45;
benign_good = and(good_hemo, hemo_bormali == 0);
mali_good = and(good_hemo, hemo_bormali == 1);
data = [hemo_all(benign_good,1)',hemo_all(benign_good,2)',hemo_all(mali_good,1)',hemo_all(mali_good,2)'];
labels = [zeros(1,length(hemo_all(benign_good,1))),1+zeros(1,length(hemo_all(benign_good,2))),2+zeros(1,length(hemo_all(mali_good,1))),3+zeros(1,length(hemo_all(mali_good,2)))];
figure;boxplot(data,labels,'Labels',{'Input benign','Output benign','Input malignant','Output malignant'})
title('Maximum hemoglobin')