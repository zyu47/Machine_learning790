%ridge_reg(trainx(:,1:7),trainy, testx(:,1:7),testy);

function [opt_beta_cv, opt_beta_t] = ridge_reg(data_x, data_y, test_x, test_y)

[m,n] = size(data_x);   
[test_m, test_n] = size(test_x); %test_n should equal to n

beta_legend = {'b1','b2','b3','b4','b5','b6','b7'};

lambda = 0:1:1000;
nlambda = max(size(lambda));

%Ridge regression

r_beta = zeros(nlambda,n);
training_error = zeros(nlambda,1);
validation_error = training_error; %10-fold cross validation
test_error = training_error;
for i = 1:nlambda,
    standard_beta = ridge(data_y,data_x,lambda(i),1);
    beta = ridge(data_y,data_x,lambda(i),0);
    r_beta(i,:) = standard_beta;
    training_error(i) = sum((data_y - [data_x,ones(m,1)]*beta).^2);
    test_error(i) = sum((test_y - [test_x,ones(test_m,1)]*beta).^2);
    %10 fold cross validation
    for j = 1:10,
        val_trainx = data_x((m/10*j-19):(m/10*j),:);
        val_trainy = data_y((m/10*j-19):(m/10*j),:);
        val_testx = data_x([1:(m/10*j-20),(m/10*j+1):200],:);
        val_testy = data_y([1:(m/10*j-20),(m/10*j+1):200],:);
        beta = ridge(val_trainy,val_trainx, lambda(i),0);
        validation_error(i) = validation_error(i) + sum((val_testy - [val_testx,ones(m*9/10,1)]*beta).^2);
    end
    validation_error(i) = validation_error(i)/10;    
end
%compute local minimum for each kind of error
[trError_min, trError_index] = min(training_error);
[vError_min, vError_index] = min(validation_error);
[teError_min, teError_index] = min(test_error);

%coefficiency calculated based on cross-validation error minimum
opt_beta_cv = ridge(data_y,data_x,lambda(vError_index));
opt_beta_t = ridge(data_y,data_x,lambda(teError_index));


subplot(2,2,1)
plot(r_beta,'LineWidth',1.5)
line([lambda(vError_index) lambda(vError_index)],ylim,'LineWidth',1.5,'LineStyle',':','color','r');
legend(beta_legend)
title('Ridge Regression')
xlabel('lambda')
ylabel('Standalized beta')

subplot(2,2,2)
plot(lambda, training_error,'LineWidth',1.5)
line([lambda(trError_index) lambda(trError_index)],ylim,'LineWidth',1.5,'LineStyle','--','color','r');
title('Training Error')
xlabel('lambda')
ylabel('RSS for training')

subplot(2,2,3)
plot(lambda, validation_error,'LineWidth',1.5)
line([lambda(vError_index) lambda(vError_index)],ylim,'LineWidth',1.5,'LineStyle','--','color','r');
title('10-fold validation')
xlabel('lambda')
ylabel('Cross validation error')

subplot(2,2,4)
plot(lambda, test_error,'LineWidth',1.5)
line([lambda(teError_index) lambda(teError_index)],ylim,'LineWidth',1.5,'LineStyle','--','color','r');
title('Testing error')
xlabel('lambda')
ylabel('RSS for testing')

%print -dpng 'ridge.png'

