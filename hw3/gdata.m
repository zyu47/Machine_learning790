% y = 0.66*x1 + 0.41 * x2 - 0.3 * x3 + error
%x1: random float number from 0 to 1;
%x2: random integer number from 1 to 5;
%x3: random float number from -2.5 to 2.5;

function [x,y] = gdata(n)

%meaningful variables
x1 = rand(n,1);
x2 = randi(5,n,1);
x3 = (rand(n,1)-0.5)*5;

%error
x4 = randn(n,1);
x5 = randn(n,1);
x6 = randn(n,1);
x7 = randn(n,1);

%variant
v1 = x1;
v2 = -x5;

y = 0.66.*x1 + 0.41.*x2 - 0.3.*x3 + randn(n,1);
x = [x1,x2,x3,x4,x5,x6,x7,v1,v2];