%load data_label.mat first
%function mvregress is used to fit linear model

decision = @(x1,x2) x2-(x1-1)^2-1;

data_sz = max(size(data));
X = horzcat(data,ones(max(size(data)),1));
coef = mvregress(X,label);
linear_decision = @(x1,x2) x1.*coef(1) + x2.*coef(2) + coef(3);

%plot linear function
step = 0:.02:2;
linear_func = (-coef(3) - coef(1).*step)/coef(2);
plot(data(1:data_sz/2,1),data(1:data_sz/2,2),'g.',data(data_sz/2+1:data_sz,1),data(data_sz/2+1:data_sz,2),'r.')
hold on
plot(step,linear_func)
hold off
print -dpng 'linear_decision.png'

%training error
linear_trainingLabel = zeros(data_sz,1);
for i = 1:data_sz,
    if linear_decision(data(i,1),data(i,2)) <0,
        linear_trainingLabel(i,1) = -1;
    else
        linear_trainingLabel(i,1) = 1;
    end
end
linear_training_error = sum(1 - label.*linear_trainingLabel)/2/data_sz;

%validation error
%Try 10-fold cross-validation here.
partition = 1:10:100;
linear_validation_error = 0;
pt = data_sz/10/2; %partition
for i = 1:10,
    linear_training_data_subset = data([1:pt*(i-1) pt*i+1:(pt*(i-1)+data_sz/2) (pt*i+data_sz/2+1):data_sz],:);
    linear_training_label_subset = label([1:pt*(i-1) pt*i+1:(pt*(i-1)+data_sz/2) (pt*i+data_sz/2+1):data_sz]);
    linear_validation_data_subset = data([(pt*(i-1)+1):pt*i (pt*(i-1)+1+data_sz/2):(pt*i+data_sz/2)],:);
    linear_validation_label_subset = label([(pt*(i-1)+1):pt*i (pt*(i-1)+1+data_sz/2):(pt*i+data_sz/2)]);
    linear_validation_label_calc = linear_validation_label_subset;
    validation_X = horzcat(linear_training_data_subset,ones(max(size(linear_training_data_subset)),1));
    validation_coef = mvregress(validation_X,linear_training_label_subset);
    linear_validation_decision = @(x1,x2) x1.*validation_coef(1) + x2.*validation_coef(2) + validation_coef(3);
    for j = 1:max(size(linear_validation_data_subset)),
        if linear_validation_decision(linear_validation_data_subset(j,1),linear_validation_data_subset(j,2)) <0,
            linear_validation_label_calc(j,1) = -1;
        else
            linear_validation_label_calc(j,1) = 1;
        end
    end
        linear_validation_error = linear_validation_error + sum(1 - linear_validation_label_subset.*linear_validation_label_calc)/2/max(size(linear_validation_label_calc));
end
linear_validation_error = linear_validation_error/10;

% test error
test_sz = 1000;
[test_data, test_true_label] = data_generation(test_sz/2);
linear_testLabel = zeros(test_sz,1);

for i = 1:test_sz,
    if linear_decision(test_data(i,1),test_data(i,2)) <0,
        linear_testLabel(i,1) = -1;
    else
        linear_testLabel(i,1) = 1;
    end
    linear_test_error = sum(1 - test_true_label.*linear_testLabel)/2/test_sz;
end




