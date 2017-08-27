%generate sample size of 10:10:100 for each group

decision = @(x1,x2) x2-(x1-1)^2-1;

linear_training_error_sz = zeros(10,1);
linear_test_error_sz = zeros(10,1);
linear_validation_error_sz = zeros(10,1);
knn_training_error = zeros(10,10);
knn_test_error = zeros(10,10);
knn_validation_error = zeros(10,10);

filename_sz = {'k1.png','k2.png','k3.png','k4.png','k5.png','k6.png','k7.png','k8.png','k9.png','k10.png'};

for sz=10:10:100,
    [data, label] = data_generation(sz);
    sorts_error_knn;
    error_linear;
    linear_training_error_sz(sz/10) = linear_training_error;
    linear_test_error_sz(sz/10) = linear_test_error;
    linear_validation_error_sz(sz/10) = linear_validation_error;
    knn_training_error(sz/10,:) = training_error;
    knn_test_error(sz/10,:) = test_error;
    knn_validation_error(sz/10,:) = validation_error;
end

%plot the graph
for k=1:10,
    plot(10:10:100,knn_validation_error(:,k),'r','linewidth',3)
    hold on
    plot(10:10:100,knn_training_error(:,k),'b','linewidth',3)
    plot(10:10:100,knn_test_error(:,k),'g','linewidth',3)
    xlabel('sample size')
    ylabel('error')
    legend('validation error','training error','test error')
    hold off
    print('-dpng',filename_sz{k})
end


plot(10:10:100,linear_validation_error_sz,'r','linewidth',3)
hold on
plot(10:10:100,linear_training_error_sz,'b','linewidth',3)
plot(10:10:100,linear_test_error_sz,'g','linewidth',3)
xlabel('sample size')
ylabel('error')
legend('validation error','training error','test error')
hold off
print('-dpng','linear_error_with_sample_size')
