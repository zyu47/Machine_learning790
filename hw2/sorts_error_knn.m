%load data_label.mat first
%error equals overall loss divided by data number

decision = @(x1,x2) x2-(x1-1)^2-1;
data_sz = max(size(data));

% training error 
training_error = zeros(1,10);
for k = 2:10, %training error is 0 when k = 1
    l = 0;  %overall raw loss.
    determined = 0; %calculate the number of points whose labels can be determined.
    for i = 1:data_sz,
        neib = knnsearch(data, data(i,:),'K',k);
        AVGlabelXY = mean(label(neib,1));
        if AVGlabelXY <0,
            determined = determined + 1;
            l = l + (1 - label(i,1) * -1);
        elseif AVGlabelXY >0,
            determined = determined + 1;
            l = l + (1 - label(i,1) * 1);
        end
    end
    training_error(k) = l/2/determined;
end

%validation error
%Try 10-fold cross-validation here. 
validation_error = zeros(1,10);
for k = 1:10,
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
    validation_error(k) = l/2/determined;
end

% test error
test_sz = 1000;
[test_data, test_true_label] = data_generation(test_sz/2);
test_error = zeros(1,10);

for k = 1:10, 
    l = 0;  %overall raw loss.
    determined = 0; %calculate the number of points whose labels can be determined.
    for i = 1:test_sz,
        neib = knnsearch(data, test_data(i,:),'K',k);
        AVGlabelXY = mean(label(neib,1));
        if AVGlabelXY <0,
            determined = determined + 1;
            l = l + (1 - test_true_label(i,1) * -1);
        elseif AVGlabelXY >0,
            determined = determined + 1;
            l = l + (1 - test_true_label(i,1) * 1);
        end
    end
    test_error(k) = l/2/determined;
end





