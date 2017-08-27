function [Rsquare, LMtraining_error, LMvalidation_error, LAtraining_error, LAvalidation_error] = part_1() 

load X1.mat
load Y1.mat

%linear model and cross-validation
[betaLM,sigma,error] = mvregress([ones(1000,1),X1],Y1); 
LMtraining_error = sum(error.^2)/1000;
Rsquare = sum(([ones(1000,1),X1]*betaLM-mean(Y1)).^2)/sum((Y1-mean(Y1)).^2);
LMvalidation_error = 0;
%10-fold cross-validation
indices = crossvalind('Kfold',1000,10);
for i = 1:10,
    test = (indices == i);
    train = ~test;
    b = mvregress([ones(sum(train),1),X1(train,:)], Y1(train,:));
    LMvalidation_error = LMvalidation_error + sum((Y1(test,:) - [ones(sum(test),1),X1(test,:)]*b).^2);
end
LMvalidation_error = LMvalidation_error/1000;

%Lasso_feature_selection
lambda = 0:0.001:0.3;
nlambda = max(size(lambda));
selection_matrix = [ones(20,1),zeros(20,19)]; %record which dimension is dropped
beta_matrix = zeros(20,nlambda); %matrix recording coefficients from lasso regression
j = 1;
for i = 1:nlambda,
    betaLA = lasso(X1,Y1,'Lambda',lambda(i));
    beta_matrix(:,i) = betaLA;
    ind = (betaLA ~= 0);
    if j <= 19 && sum(ind) ~= sum(selection_matrix(:,j)),
        selection_matrix(:,j+1) = ind;
        j = j+1;
    end
end

LAtraining_error = zeros(1,21);
LAvalidation_error = zeros(1,21);
LAvalidation_error(21) = sum((Y1-mean(Y1)).^2)/1000; %error when using 0 dimension
LAtraining_error(21) = LAvalidation_error(21);
for j = 1:20, %20 different selections depending which dimension to drop
    select = logical(selection_matrix(:,j));
    x = X1(:,select);
    [b,s,e] = mvregress([ones(1000,1),x],Y1); 
    LAtraining_error(j) = sum(e.^2)/1000;
    indices = crossvalind('Kfold',1000,10);
    for i = 1:10,
        test = (indices == i);
        train = ~test;
        b = mvregress([ones(sum(train),1),x(train,:)], Y1(train,:));
        LAvalidation_error(j) = LAvalidation_error(j) + sum((Y1(test,:) - [ones(sum(test),1),x(test,:)]*b).^2);
    end
    LAvalidation_error(j) = LAvalidation_error(j)/1000;
end


