load X2.mat
load Y2.mat

%linear model
[b,s,e] = mvregress([ones(101,1),X2'],Y2');
plot(X2,Y2)
hold on
plot(0:0.01:1, [ones(101,1),(0:0.01:1)']*b,'r');
hold off


knots = [0.3 0.8];
left = X2 <=knots(1);
middle = (knots(1) < X2 & X2 <=knots(2));
right = (X2 >knots(2));

[b1,s1,e1] = mvregress([ones(sum(left),1), X2(left)'],Y2(left)');
[b2,s2,e2] = mvregress([ones(sum(middle),1), X2(middle)'],Y2(middle)');
[b3,s3,e3] = mvregress([ones(sum(right),1), X2(right)'],Y2(right)');

plot(X2,Y2,'.')
hold on
plot(0:0.01:knots(1), (0:0.01:knots(1)).*b1(2) + b1(1))
plot(knots(1):0.01:knots(2), (knots(1):0.01:knots(2)).*b2(2) + b2(1))
plot(knots(2):0.01:1, (knots(2):0.01:1).*b3(2) + b3(1))
hold off

plot(X2(left),e1,'.')
hold on
plot(X2(middle),e2,'.')
plot(X2(right),e3,'.')
hold off


%continuous piecewise
x = [ones(101,1),X2',zeros(101,2)];
for i = 1:101,
    x(i,3) = max(0,x(i,2) - 0.3);
    x(i,4) = max(0,x(i,2) - 0.8);
end
[b4,s4,e4] = mvregress(x,Y2');
test = [ones(101,1),(0:0.01:1)',zeros(101,2)];
for i = 1:101,
    test(i,3) = max(0,test(i,2) - 0.3);
    test(i,4) = max(0,test(i,2) - 0.8);
end
plot(X2,Y2,'.')
hold on
plot(0:0.01:1, test*b4,'r')
hold off



