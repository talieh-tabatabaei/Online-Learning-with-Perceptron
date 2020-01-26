function [SV,G,Y,Index,hat_y_t] = VBP(x_t,y_t,index,SV,G,Y,Index,ker,p1,p2,C)
%Online budget perceptron algorithm with variable size of B (variable BP)
% INPUT:
%      y_t:         class label of t-th instance;
%      x_t:         t-th training data instance, e.g., X(:,t) (colomn vector);
%      index:       index of sample (t)
%      SV:          previous SV set
%      G:           Matrix containign gamma values for SV's
%      Y:           Matrix containing labels for SV's
%      Index:       Matrix containing indexes of SV's in the stream
%      C:           regularization parameter
%      ker:         kernel type
%      p1 & p2:     kernel paramenters is applicable. otherwise []
%            
%          Values for ker: 'linear'     - 
%                   'delta'      -  
%                   'poly'       - p1 is degree of polynomial
%                   'rbf'        - p1 is width of rbfs (sigma)
%                   'sigmoid'    - p1 is scale, p2 is offset
%                   'spline'     -
%                   'bspline'    - p1 is degree of bspline
%                   'fourier'    - p1 is degree
%                   'erfb'       - p1 is width of rbfs (sigma)
%                   'anova'      - p1 is max order of terms


%OUTPUT:
%      SVs:         set of support vectors


%% Initialization

N=size(SV,2)+1;  %N is the counter for SV's

%% calculating the predicted values in RHKS
if index==1
    gamma_t=0.1; 
    SV(:,N)=x_t;         %putting the first sample in the SV list 
    Y(N,1)=y_t;          %Y is the matrix containing labels of the SV's 
    G(N,1)=gamma_t;      %G is the matrix containing the gamma values for each SV
    Index(N,1)=index;    %Index is the matrix containing indexes of the SV's in the sequence
end
hat_y_t=y_t;
%Calculating k(xi,xj), where xi's are those samples in the SV set
if index>1
   for l=1:size(SV,2)
       for j=1:size(x_t,2)
           K(l,j) = kernel(ker,SV(:,l),x_t(:,j),p1,p2);          
       end
   end
   f_t=sum(G.*Y.*K);  %multiplying each kernel with proper label and gamma
   %% Prediction for kernel PA
   hat_y_t=sign(f_t);
   if (hat_y_t == 0) % binary prediction
       hat_y_t = 1;
   end
   
%% Making Update to the SV set
   s_t = norm(x_t)^2;
   l_t = max(0,1-y_t*f_t); %Hing loss
   if (l_t > 0)
       SV(:,N)=x_t;         %putting the sample in the SV list
%      gamma_t = min(C,l_t/s_t); % PA-I
%      gamma_t = l_t/(s_t+1/(2*C)); % PA-II
       gamma_t=1;           %as in original BP
       Y(N,1)=y_t;          %Y is the matrix containing labels of the SV's 
       G(N,1)=gamma_t;      %G is the matrix containing the gamma values for each SV
       Index(N,1)=index;    %Index is the matrix containing indexes of the SV's in the sequence
       [SV,Y,G] = DC(SV,Y,G);
   end

end

end


%% DistillCache function
function [SV2,Y2,G2] = DC(SV,Y,G)
%DistillCache function checks for the potential removal of SV's
SV2=SV;
Y2=Y;
G2=G;
N=size(SV,2);
delete=0;
for i=1:N
    i=i-delete; %according to the number of samples which has been removed from the SV set, the counter index goes back (delete is the counter for the number of removed samples )
    SV=SV2;
    Y=Y2;
    G=G2;
    x_t= SV(:,i);
    y_t=Y(i,1);
    SV(:,i)=[];
    G(i,:)=[];
    Y(i,:)=[];
    for l=1:size(SV,2)
       for j=1:size(x_t,2)
           K(l,j) = kernel('rbf',SV(:,l),x_t(:,j),0.2,[]);          
       end
    end
    f_t=sum(G.*Y.*K);  %multiplying each kernel with proper label and gamma
    hat_y_t=sign(f_t);
    if (hat_y_t == 0) % binary prediction
        hat_y_t = 1;
    end
    l_t = max(0,1-y_t*f_t); %Hing loss
    if l_t==0
        SV2(:,i)=[];
        Y2(i,:)=[];
        G2(i,:)=[];
        N=N-1;
        K=[];
        delete=delete+1; %This is because when a sample is removed from SV set, the counter index should move one position back. 
    end
end
end

