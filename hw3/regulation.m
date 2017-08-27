function regulation(data_x, data_y)

[m,n] = size(data_x);
beta_legend = {'b1','b2','b3','b4','b5','b6','b7'};

%LASSO

lambda = 0:.001:0.5;

l_beta = zeros(max(size(lambda)),n);
for i = 1:max(size(lambda)),
    l_beta(i,:) = (lasso(data_x,data_y, 'Lambda',lambda(i)))';
end

subplot(1,2,1)
plot(lambda,l_beta,'LineWidth',1.5)
legend(beta_legend)
title('Lasso Regression')
xlabel('lambda')
ylabel('beta')

%RIDGE
lambda = 0:0.1:1000;
r_beta = zeros(max(size(lambda)),n);
for i = 1:max(size(lambda)),
    r_beta(i,:) = (data_x'*data_x + lambda(i).*eye(n))\(data_x'*data_y);
end

subplot(1,2,2)
plot(lambda, r_beta, 'LineWidth',1.5)
legend(beta_legend)
title('Ridge Regression')
xlabel('lambda')
ylabel('beta')


