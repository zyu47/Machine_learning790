function bruteforce(data_x, data_y, test_x, test_y)

[m,n] = size(data_x);
[test_m, test_n] = size(test_x); %test_n should equal to n

error0 = (data_y - mean(data_y))' * (data_y - mean(data_y)); %RSS when beta equals 0
test_error0 = (test_y - mean(test_y))' * (test_y - mean(test_y)); %test RSS when beta equals 0
validation_error0 = 0;
for j = 1:10,
    val_y = data_y([1:(m/10*j-20),(m/10*j+1):200],:);
    validation_error0 = validation_error0 + (val_y - mean(val_y))'*(val_y - mean(val_y));
end
validation_error0 = validation_error0/10;
subplot(1,3,1);
plot(0,error0,'r.','markersize',15); %plot for training data
hold on
subplot(1,3,3);
plot(0,test_error0,'r.','markersize',15); %plot for test data
hold on
subplot(1,3,2);
plot(0,validation_error0,'r.','markersize',15); %plot for test data
hold on


for k = 1:n,
    index = nchoosek(1:n,k);
    enumber = nchoosek(n,k); %the number of possible choices
    train_error = zeros(enumber,1);
    test_error = train_error;
    validation_error = train_error;
    for i = 1:enumber,
        [beta,sigma,e] = mvregress([data_x(:,index(i,:)),ones(m,1)], data_y); %add one column of 1 for calculating intercept
        train_error(i) = sum(e.^2);
        test_error(i) = sum((test_y - [test_x(:,index(i,:)),ones(test_m,1)]*beta).^2);
        %compute validation_error
        for j = 1:10,
            val_trainx = data_x((m/10*j-19):(m/10*j),:);
            val_trainy = data_y((m/10*j-19):(m/10*j),:);
            val_testx = data_x([1:(m/10*j-20),(m/10*j+1):200],:);
            val_testy = data_y([1:(m/10*j-20),(m/10*j+1):200],:);
            [beta,sigma,e] = mvregress([val_trainx(:,index(i,:)),ones(m/10,1)], val_trainy); %add one column of 1 for calculating intercept
            validation_error(i) = validation_error(i) + sum((val_testy - [val_testx(:,index(i,:)),ones(m*9/10,1)]*beta).^2);
        end
        validation_error(i) = validation_error(i)/10;
        %compute validation_error end
    end
    
    subplot(1,3,1)
    plot(k,train_error,'g.', 'markersize',15);
    plot(k, min(train_error),'r.','markersize',15);
    
    subplot(1,3,3)
    plot(k,test_error,'g.', 'markersize',15);
    plot(k, min(test_error),'r.','markersize',15);
    
    subplot(1,3,2)
    plot(k,validation_error,'g.', 'markersize',15);
    plot(k, min(validation_error),'r.','markersize',15);
    
end

subplot(1,3,1)
xlabel('Subset Size k')
ylabel('Residual Sum-of-Squares')
xlim([-0.5,n+0.5])
title('Training Data')
hold off
subplot(1,3,3)
xlabel('Subset Size k')
ylabel('Residual Sum-of-Squares')
xlim([-0.5,n+0.5])
title('Test Data')
hold off
subplot(1,3,2)
xlabel('Subset Size k')
ylabel('Residual Sum-of-Squares')
xlim([-0.5,n+0.5])
title('10-fold cross validation')
hold off


%print -dpng 'bruteforce.png'