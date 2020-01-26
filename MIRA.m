function [w] = MIRA(x_t,y_t,index,model)
%Online margin infused relaxed algorithm (MIRA)
% INPUT:
%      y_t:     class label of t-th instance;
%      x_t:     t-th training data instance, e.g., X(:,t) (colomn vector);
%      index:   index of sample (t)
%      model:   previous w(row vector)

%OUTPUT:
%      w1:   weight vector for PAI
%      w2:   weight vector for PAII
%      SV: indexes corresponding to support vectors


%% Initialization
w = model;

%% Prediction
f_t =  w*x_t;
hat_y_t = sign(f_t);
if (hat_y_t == 0) % binary prediction
    hat_y_t = 1;
end


%% Making Update
s_t      = norm(x_t)^2;
l_t = max(0,1-y_t*f_t); %Hing loss
% l_t = (hat_y_t ~= y_t); % 0: correct prediction, 1: incorrect
if (l_t > 0)
% if (l_t==1)
    gamma_t = -(y_t*f_t)/s_t 
    if gamma_t<0
        alpha_t=0;
    elseif 0<=gamma_t<=1
        alpha_t=gamma_t;
    else alpha_t=1;
    end
    alpha_t=alpha_t
    w       = w + alpha_t*y_t*x_t';
end


end

