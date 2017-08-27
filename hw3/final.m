bruteforce(trainx(:,1:3),trainy, testx(:,1:3),testy);
print -dpng 'bruteforce3.png'
[bl3] = lasso_reg(trainx(:,1:3),trainy, testx(:,1:3),testy);
print -dpng 'lasso3.png'
[br3c, br3t] = ridge_reg(trainx(:,1:3),trainy, testx(:,1:3),testy);
print -dpng 'ridge3.png'

bruteforce(trainx(:,1:7),trainy, testx(:,1:7),testy);
print -dpng 'bruteforce7.png'
[bl7] = lasso_reg(trainx(:,1:7),trainy, testx(:,1:7),testy);
print -dpng 'lasso7.png'
[br7c, br7t] = ridge_reg(trainx(:,1:7),trainy, testx(:,1:7),testy);
print -dpng 'ridge7.png'

bruteforce(trainx(:,1:8),trainy, testx(:,1:8),testy);
print -dpng 'bruteforce8.png'
[bl8] = lasso_reg(trainx(:,1:8),trainy, testx(:,1:8),testy);
print -dpng 'lasso8.png'
[br8c, br8t] = ridge_reg(trainx(:,1:8),trainy, testx(:,1:8),testy);
print -dpng 'ridge8.png'

bruteforce(trainx(:,[1:7,9]),trainy, testx(:,[1:7,9]),testy);
print -dpng 'bruteforce9.png'
[bl9] = lasso_reg(trainx(:,[1:7,9]),trainy, testx(:,[1:7,9]),testy);
print -dpng 'lasso9.png'
[br9c, br9t] = ridge_reg(trainx(:,[1:7,9]),trainy, testx(:,[1:7,9]),testy);
print -dpng 'ridge9.png'

bruteforce([trainx(:,1:3),trainx(:,4:7)*20],trainy, [testx(:,1:3),testx(:,4:7)*20],testy);
print -dpng 'bruteforceLE.png'
[bl7LE] = lasso_reg([trainx(:,1:3),trainx(:,4:7)*20],trainy, [testx(:,1:3),testx(:,4:7)*20],testy);
print -dpng 'lassoLE.png'
[br7cLE, br7tLE] = ridge_reg([trainx(:,1:3),trainx(:,4:7)*20],trainy, [testx(:,1:3),testx(:,4:7)*20],testy);
print -dpng 'ridgeLE.png'