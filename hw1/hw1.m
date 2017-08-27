%generate points and plot
sz = 100; %number of points for each group
xn = [-1.+2*rand(sz,2),ones(sz,1)*-1]; %-1, red
xp = [-1.+2*rand(sz,2),ones(sz,1)]; %+1, green
plot(xn(:,1),xn(:,2),'r.',xp(:,1), xp(:,2),'g.')

%compute 0-1 loss
lossmap = zeros(max(size(-1:.01:1))^2, 3);
r = 1;
for w1 = -1:.01:1
    for w2 = -1:.01:1
        x = [xn;xp]*[w1 0; w2 0; 0 1]; %calculate w1*x1+w2*x2 supposing b = 0
        sorted_x = sortrows(x,1);
        loss_start = sz; %choose a small enough b to make every point +1
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

%plot using mesh
x = -1:.01:1;
y = -1:.01:1;
[x,y] = meshgrid(x,y);
z = vec2mat(lossmap(:,3),201);
% colorm = [zeros(40401,1), lossmap(:,3), zeros(40401,1)]
%scatter(lossmap(:,1),lossmap(:,2),[],colorm.*0.01, 'filled','s')