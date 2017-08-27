%optimal decision function: f(x1,x2) = x2-(x1-1)^2-1
%data probability distribution: for negative data, 90% chance in 
    %f(x1,x2)<0, 10% chance in f(x1,x2) >0; vice versa for positive data
%data range (0,2) for both x1 and x2
%data for each group have sz = 100 labels
%the training data and label with decision anonymous function are saved in
    %a file called data_label.mat; load this before proceed with analysis

function [data,label] = data_generation(sz)
p = 0.1;
decision = @(x1,x2) x2-(x1-1)^2-1;

%generate data and label
dataP = zeros(sz,2); % positive data
dataN = dataP;
iP = 1;
iN = 1;
while iP <= sz || iN <= sz,
    tmp = rand(1,3).*2;
    if decision(tmp(1),tmp(2)) <0,
        if tmp(3) < 2*p && iP <= sz,
            dataP(iP,:) = tmp(1:2);
            iP = iP +1;
        end
        if tmp(3) >= 2*p && iN <= sz,
            dataN(iN,:) = tmp(1:2);
            iN = iN +1;
        end
    else
        if tmp(3) >= 2*p && iP <= sz,
            dataP(iP,:) = tmp(1:2);
            iP = iP +1;
        end
        if tmp(3) < 2*p && iN <= sz,
            dataN(iN,:) = tmp(1:2);
            iN = iN +1;
        end
    end
end
%plot data
%plot(dataP(:,1),dataP(:,2),'g.',dataN(:,1),dataN(:,2),'r.')

data = vertcat(dataP, dataN);
label = vertcat(ones(sz,1), -1.*ones(sz,1));










