function [w1 w2 SV] = PA(x_t,y_t,index,model1,model2,C, SV)
%Online passive aggressive algorithm
% INPUT:
%      y_t:     class label of t-th instance;
%      x_t:     t-th training data instance, e.g., X(:,t) (colomn vector);
%      index:   index of sample (t)
%      model1&2:   previous w1&2 for PAI&II (row vector)
%      C:       regularization parameter

%OUTPUT:
%      w1:   weight vector for PAI
%      w2:   weight vector for PAII
%      SV:   number of support vectors


%% Initialization
w1 = model1;
w2 = model2;
C=C;
%% Prediction for PAI
f1_t =  w1*x_t;
hat_y1_t = sign(f1_t);
if (hat_y1_t == 0) % binary prediction
    hat_y1_t = 1;
end

%% Prediction for PAII
f2_t =  w2*x_t;
hat_y2_t = sign(f2_t);
if (hat_y2_t == 0) % binary prediction
    hat_y2_t = 1;
end


%% Making Update for PAI
s_t      = norm(x_t)^2;
l1_t = max(0,1-y_t*f1_t); %Hing loss
if (l1_t > 0)
    SV=SV+1;
    gamma1_t = min(C,l1_t/s_t); % PA-I
    w1      = w1 + gamma1_t*y_t*x_t';
end

%% Making Update for PAII
l2_t = max(0,1-y_t*f2_t); %Hing loss
if (l2_t > 0)
    gamma2_t = l2_t/(s_t+1/(2*C)); % PA-II
    w2       = w2 + gamma2_t*y_t*x_t';
end


end


