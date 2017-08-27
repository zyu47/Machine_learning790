%load data_label.mat first

step = 0:.02:2;
stepsz = max(size(step));
decisionlabel = meshgrid(step,step);
data_sz = max(size(data));

cmap = [1 0.8 0.8; 0.95 1 0.95; 0.9 0.9 1];
colormap(cmap);
filename = {'1.png','2.png','3.png','4.png','5.png','6.png','7.png','8.png','9.png','10.png'};

%plot k-NN decision boundary
for k = 1:10,
    for x = 1:stepsz,
        for y = 1:stepsz,
            neib = knnsearch(data, [step(x),step(y)],'K',k);
            AVGlabelXY = mean(label(neib,1));
            if AVGlabelXY <0,
                decisionlabel(y,x) = -1;
            elseif AVGlabelXY >0,
                decisionlabel(y,x) = 1;
            else
                decisionlabel(y,x) = 0;
            end
        end
    end
    imagesc(step,step,decisionlabel)
    hold on
    set(gca,'YDir','normal')
    plot(data(1:data_sz/2,1),data(1:data_sz/2,2),'g.',data(data_sz/2+1:data_sz,1),data(data_sz/2+1:data_sz,2),'r.')
    hold off
    print('-dpng',filename{k})
end


