function [w SV] = perceptron(x_t,y_t,index,model,SV)
%Online perceptron algorithm
% INPUT:
%      y_t:     class label of t-th instance;
%      x_t:     t-th training data instance, e.g., X(:,t) (colomn vector);
%      index:   index of sample (t)
%      model:   previous w (row vector)
%      SV:      counter for number of SV's

%OUTPUT:
%      w:   weight vector
%      SV: indexes corresponding to support vectors


%% Initialization
w = model;

%% Prediction
f_t = w*x_t;
hat_y_t = sign(f_t);
if (hat_y_t == 0) % binary prediction
    hat_y_t = 1;
end

%% Making Update
if index==1
    i=1;
end
l_t = (hat_y_t ~= y_t); % 0: correct prediction, 1: incorrect
if (l_t==1)
    SV=SV+1;
    w = w + y_t*x_t';
%     SV(1,i)=index;
%     i=i+1;
end

end
