%lasso_reg(trainx(:,1:7),trainy, testx(:,1:7),testy);

function opt_beta = lasso_reg(data_x, data_y, test_x, test_y)

[m,n] = size(data_x);
[test_m, test_n] = size(test_x); %test_n should equal to n

beta_legend = {'b1','b2','b3','b4','b5','b6','b7'};

lambda = 0:.005:0.8;
nlambda = max(size(lambda));

%LASSO training and test

l_beta = zeros(nlambda,n); %record coefficents with different lambda
training_error = zeros(nlambda,1);
validation_error = training_error; %10-fold cross validation
test_error = training_error;
for i = 1:nlambda,
    [beta, info] = lasso(data_x,data_y, 'Lambda',lambda(i));
    beta0 = getfield(info,'Intercept');
    l_beta(i,:) = beta;
    training_error(i) = sum((data_y - [data_x,ones(m,1)]*[beta;beta0]).^2);
    test_error(i) = sum((test_y - [test_x,ones(test_m,1)]*[beta;beta0]).^2);
    %10 fold cross validation
    for j = 1:10,
        val_trainx = data_x((m/10*j-19):(m/10*j),:);
        val_trainy = data_y((m/10*j-19):(m/10*j),:);
        val_testx = data_x([1:(m/10*j-20),(m/10*j+1):200],:);
        val_testy = data_y([1:(m/10*j-20),(m/10*j+1):200],:);
        [beta, info] = lasso(val_trainx,val_trainy, 'Lambda',lambda(i));
        beta0 = getfield(info,'Intercept');
        validation_error(i) = validation_error(i) + sum((val_testy - [val_testx(:,1:n),ones(m*9/10,1)]*[beta;beta0]).^2);
    end
    validation_error(i) = validation_error(i)/10;
end
%compute local minimum for each kind of error
[trError_min, trError_index] = min(training_error);
[vError_min, vError_index] = min(validation_error);
[teError_min, teError_index] = min(test_error);

%coefficiency calculated based on cross-validation error minimum
opt_beta = lasso(data_x,data_y,'Lambda',lambda(vError_index));

subplot(2,2,1)
plot(lambda,l_beta,'LineWidth',1.5)
line([lambda(vError_index) lambda(vError_index)],ylim,'LineWidth',1.5,'LineStyle',':','color','r');
legend(beta_legend)
title('Lasso Regression Coefficient')
xlabel('lambda')
ylabel('Coefficient')

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

%print -dpng 'lasso.png'


