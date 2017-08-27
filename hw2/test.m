decision = @(x1,x2) x2-(x1-1)^2-1;
data_sz = max(size(data));

for k = 3,
    l = 0;  %overall raw loss.
    determined = 0; %calculate the number of points whose labels can be determined.
    pt = data_sz/10/2; %partition
    for i = 1:10,
        training_data_subset = data([1:pt*(i-1) pt*i+1:(pt*(i-1)+data_sz/2) (pt*i+data_sz/2+1):data_sz],:);
        training_label_subset = label([1:pt*(i-1) pt*i+1:(pt*(i-1)+data_sz/2) (pt*i+data_sz/2+1):data_sz]);
        validation_data_subset = data([(pt*(i-1)+1):pt*i (pt*(i-1)+1+data_sz/2):(pt*i+data_sz/2)],:);
        validation_label_subset = label([(pt*(i-1)+1):pt*i (pt*(i-1)+1+data_sz/2):(pt*i+data_sz/2)]);
        for j = 1:max(size(validation_data_subset)),
            neib = knnsearch(training_data_subset, validation_data_subset(j,:),'K',k);
            AVGlabelXY = mean(training_label_subset(neib,1));
            if AVGlabelXY <0,
                determined = determined + 1;
                l = l + (1 - validation_label_subset(j,1) * -1);
            elseif AVGlabelXY >0,
                determined = determined + 1;
                l = l + (1 - validation_label_subset(j,1) * 1);
            end
        end
    end
    validation_error1 = l/2/determined;
end

load data_label.mat
validation_error2 = 0;
for k = 3,
    partition = 1:10:100;
    for i = partition,
        training_data_subset = data([1:i-1 i+10:100 101:i+99 i+110:200],:);
        training_label_subset = label([1:i-1 i+10:100 101:i+99 i+110:200]);
        validation_data_subset = data([i:i+9 i+100:i+109],:);
        validation_label_subset = label([i:i+9 i+100:i+109]);
        validation_label_calc = validation_label_subset;
        for j = 1:max(size(validation_data_subset)),
            neib = knnsearch(training_data_subset, validation_data_subset(j,:),'K',k);
            AVGlabelXY = mean(training_label_subset(neib,1));
            if AVGlabelXY <0,
                validation_label_calc(j,1) = -1;
            else
                validation_label_calc(j,1) = 1;
            end
        end
        validation_error2 = validation_error2 + sum(1 - validation_label_subset.*validation_label_calc)/max(size(validation_label_calc));
    end
    validation_error2 = validation_error2/2/max(size(partition));
end