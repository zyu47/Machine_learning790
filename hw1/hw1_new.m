%generate points and plot
sz = 100; %number of points for each group
xn = [rand(sz,2) * 0.8,ones(sz,1)*-1]; %-1, red; shift the mean of negative examples
xp = [rand(sz,2),ones(sz,1)]; %+1, green
%plot(xn(:,1),xn(:,2),'r.',xp(:,1), xp(:,2),'g.')

step = [-1:.01:-0.01, 0.01:.01:1];
lossmap = zeros(max(size(step))^2, 3); %result matrix
r = 1; %

%compute 0-1 loss
%we don't need to specifically compute b since we don't need b for calculating loss
for w1 = step
    for w2 = step
        x = [xn;xp]*[w1 0; w2 0; 0 1]; %calculate w1*x1+w2*x2
        sorted_x = sortrows(x,1);
        loss_start = sz; %choose a small enough b to make every point in group of +1
        loss_x = zeros(1,2*sz); %matrix to record loss value at each point
        if sorted_x(1,2) == -1  %base case (if the decision boundary moves beyond the first point)
            loss_x(1) = loss_start - 1;
        else
            loss_x(1) = loss_start + 1;
        end
        for i = 2:2*sz %go through every point, calculate loss if the decision boundary is moved beyond each point
            if sorted_x(i,2) == -1
                loss_x(i) = loss_x(i-1) - 1;
            else
                loss_x(i) = loss_x(i-1) + 1;
            end
        end
        lossmap(r,1) = w1;
        lossmap(r,2) = w2;
        lossmap(r,3) = min(loss_x);
        r = r+1;
    end
end

%compute hinge loss I
%first method is to specify b by w1*x1+w2*x2 each time and calculate loss
%dynamically, faster method 
for w1 = step
    for w2 = step
        xn_calc = xn*[w1; w2; 0]; %calculate w1*x1+w2*x2 for negative group
        xp_calc = xp*[w1; w2; 0];
        xn_sort = sort(xn_calc);
        xp_sort = sort(xp_calc);
        loss_p = 0; %positive group loss; starting from f(x) > 1 for all x
        loss_n = sz * (1 + (1 - xp_sort(1))) + sum(xn_calc); % 1 - xp_sort(1) is the largest possible b
        loss_min = loss_n + loss_p;
        for i = 2:sz %calculate loss when b > -1
            %b = 1 - xp_sort(i);
            loss_p = loss_p + (i-1)*(xp_sort(i) - xp_sort(i-1));
            loss_n = loss_n - sz * (xp_sort(i) - xp_sort(i-1));
            loss_tmp = loss_p + loss_n;
            if loss_tmp < loss_min
                loss_min = loss_tmp;
            end
        end
        %when b < -1, starting from b = -1 - xn_sort(1)
        loss_p = sz * (1 - (-1 - xn_sort(1))) - sum(xp_calc);
        loss_n = sz * (1 + (-1 - xn_sort(1))) + sum(xn_calc);
        for i = 2:sz
            %b = -1 - xn_sort(i);
            loss_p = loss_p + sz * (xn_sort(i) - xn_sort(i-1));
            loss_n = loss_n - (sz - i + 1) * (xn_sort(i) - xn_sort(i-1));
            loss_tmp = loss_p + loss_n;
            if loss_tmp < loss_min
                loss_min = loss_tmp;
            end
        end
        lossmap(r,1) = w1;
        lossmap(r,2) = w2;
        lossmap(r,3) = loss_min;
        r = r+1;
    end
end



%compute hinge loss II
%second method is to specify b by decreasing .01 each time and calculate loss, slower method but gives the same result
for w1 = step
    for w2 = step
        xn_calc = xn*[w1; w2; 0]; %calculate w1*x1+w2*x2 for negative group
        xp_calc = xp*[w1; w2; 0];
        b_max = 1-min(xp_calc)+0.01; % max possible b
        b_min = -1-max(xn_calc)-0.01; % min possible b
        min_loss = 200;
        for b = b_min:.01:b_max
            loss_tmp = sum(max(0, 1+xn_calc+b))+sum(max(0, 1-xp_calc-b));
            if loss_tmp < min_loss
                min_loss = loss_tmp;
            end
        end
        lossmap(r,1) = w1;
        lossmap(r,2) = w2;
        lossmap(r,3) = min_loss;
        r = r+1;
    end
end

%plot using surf or scatter
x = step;
y = step;
[x,y] = meshgrid(x,y);
z = vec2mat(lossmap(:,3),max(size(step)));
surf(x,y,z)

%scatter(lossmap(:,1),lossmap(:,2),[],lossmap(:,3).*0.01, 'filled','s')
