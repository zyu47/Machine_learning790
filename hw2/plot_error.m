plot(1:10,validation_error,'r')
hold on
plot(1:10,training_error,'b')
plot(1:10,test_error,'g')
plot(5,linear_validation_error,'rd','MarkerFaceColor','r')
plot(5,linear_training_error,'bd','MarkerFaceColor','b')
plot(5,linear_test_error,'gd','MarkerFaceColor','g')
xlabel('k')
ylabel('error')
legend('validation error','training error','test error')
hold off
print '-dpng' 'error.png'