%read data
data = csvread('train.csv', 1,1);
label = data(:,55);
data = data(:,1:54);
test_data = csvread('test.csv', 1,1);

%response = {'Spruce/Fir','Lodgepole Pine','Ponderosa Pine','Cottonwood/Willow','Aspen','Douglas-fir','Krummholz'};

%view(b,'Mode','Graph')

%csvwrite('res.csv',res)

%Average Hillshade
for i = 1:15120,
    for j = 1: 6,
        data_avg(i,j) = data(i,j);
    end
    data_avg(i,7) = mean(data(i,7:9));
    %data_new(i,8) = data(i,10);
    %data_new(i,11) = 1*test_data(i,11) + 2*test_data(i,12) + 3*test_data(i,13) + 4*test_data(i,14);
    for j = 8:52,
        data_avg(i,j) = data(i,j+2);
    end
end

%Collapse binary data
for i = 1:15120,
    for j = 1: 10,
        data_coll(i,j) = data(i,j);
    end
   % data_avg(i,7) = mean(data(i,7:9));
    data_coll(i,11) = data(i,11:14)*(1:4)';
    data_coll(i,12) = data(i,15:54)*(1:40)';
end

%Average Hillshade
for i = 1:565892,
    for j = 1: 6,
        test_data_avg(i,j) = test_data(i,j);
    end
    test_data_avg(i,7) = mean(test_data(i,7:9));
    %data_new(i,8) = data(i,10);
    %data_new(i,11) = 1*test_data(i,11) + 2*test_data(i,12) + 3*test_data(i,13) + 4*test_data(i,14);
    for j = 8:52,
        test_data_avg(i,j) = test_data(i,j+2);
    end
end

%Collapse binary data
for i = 1:565892,
    for j = 1: 10,
        test_data_coll(i,j) = test_data(i,j);
    end
   % data_avg(i,7) = mean(data(i,7:9));
    test_data_coll(i,11) = test_data(i,11:14)*(1:4)';
    test_data_coll(i,12) = test_data(i,15:54)*(1:40)';
end

%full_data_new = [data_avg(:,1:8),data(:,7:9),data_coll(:,11:12),label];
%test_data_new = [test_data_avg(:,1:8),test_data(:,7:9),test_data_coll(:,11:12)];
%decision tree and crossvalidation
tree = fitctree(data,label);
for i = 2:1001,
    tree_small = fitctree(data,label,'MaxNumSplits',i,'CrossVal','on');
    cv_error(i-1) = kfoldLoss(tree_small);
end
fitctree(data,label,'MaxNumSplits',5,'Predictornames',{'Elevation','Aspect','Slope','HdtHydro','VdtHydro','HdtRoad','HS9','HSN','HS3','HdtFP','W1','W2','W3','W4','ST1','ST2','ST3','ST4','ST5','ST6','ST7','ST8','ST9','ST10','ST11','ST12','ST13','ST14','ST15','ST16','ST17','ST18','ST19','ST20','ST21','ST22','ST23','ST24','ST25','ST26','ST27','ST28','ST29','ST30','ST31','ST32','ST33','ST34','ST35','ST36','ST37','ST38','ST39','ST40'});

%colormatrix
c = [0 0 1;0 1 0;1 0 0;1 1 0;1 0 1;0 1 1;0 0 0];
cmatrix = zeros(15120,3);
for i = 1:15120,
    j = label(i);
    cmatrix(i,:) = c(j,:);
end

cv_error_importance = zeros(54,1);
for i = 1:54,
    tree_no_onefeature = fitctree(data(:,[1:i-1,i+1:54]),label,'CrossVal','on');
    cv_error_importance(i) = kfoldLoss(tree_no_onefeature);
end