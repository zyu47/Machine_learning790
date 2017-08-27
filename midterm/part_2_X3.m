% f(X) = h1i(X1) + h2j(X2)

load X3
load Y3

x1 = [X3(:,1),zeros(10000,4)];
x2 = [X3(:,2),zeros(10000,4)];
for i = 2:5,
    for j = 1:10000,
        x1(j,i) = max(0,x1(j,1)-i+1);
        x2(j,i) = max(0,x2(j,1)-i+1);
    end
end

[b,s,e] = mvregress([ones(10000,1),x1,x2],Y3);
yhat = zeros(100,100);
step = 0.01:0.05:5;
%xmatrix = meshgrid(0.01:0.05:5);
for i = 1:100,
    for j = 1:100,
        yhat(i,j) = b(1) + b(2)*step(i) + b(3) * max(0,step(i)-1) + b(4) * max(0,step(i)-2)+ b(5) * max(0,step(i)-3)+ b(6) * max(0,step(i)-4)...
            + b(7) * step(j)+ b(8) * max(0,step(j)-1)+ b(9) * max(0,step(j)-2)+ b(10) * max(0,step(j)-3)+ b(11) * max(0,step(j)-4);
    end
end

scatter(X3(:,1),X3(:,2),[],,'filled')
scatter3(X3(:,1),X3(:,2),e,'filled')

%%test
[b,s,e] = mvregress([ones(10000,1),x1],Y3);
plot(X3(:,1),Y3)
hold on
plot(X3(:,1),Y3-e)
hold off
yhat = 0.01:0.05:5;
xt = [ones(100,1),zeros(100,5)];
xt(:,2) = (0.01:0.05:5)';
for i = 3:6,
    for j = 1:100,
        xt(j,i) = max(0,xt(j,2)-i+1);
    end
end
yhat = (xt*b)';
plot(0.01:0.05:5,yhat)