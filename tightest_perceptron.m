function [SV,G,Y,Index,hat_y_t,pos,neg,K2,t] = tightest_perceptron(x_t,y_t,index,SV,G,Y,Index,ker,p1,p2,C,B,pos,neg,K2)
%Online tightest budget perceptron algorithm (Zhuang Wang and Slobodan Vucetic, Tighter Perceptron with 
% Improved Dual Use of Cached Data for Model Representation and Validation, Proceedings of International Joint 
% Conference on Neural Networks, Atlanta, 2009. )
% INPUT:
%      y_t:         class label of t-th instance;
%      x_t:         t-th training data instance, e.g., X(:,t) (colomn vector);
%      index:       index of sample (t)
%      SV:          previous SV set
%      G:           Matrix containign gamma values for SV's
%      Y:           Matrix containing labels for SV's
%      Index:       Matrix containing indexes of SV's in the stream
%      C:           regularization parameter for soft margin
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
%      B:           budget size  
%      pos,neg:     Matrices containing number of positive and negative neigbours for each SV 
%      K2:          Kernel matrix over SV set

%OUTPUT:
%      SVs:         set of support vectors
%      Index:       index of SV's in the stream
%      G:           Matrix containign gamma values for SV's
%      Y:           Matrix containing labels for SV


%% Initialization

N=size(SV,2)+1;  %N is the counter for SV's

%% calculating the predicted values in RHKS
if index==1
    neg=zeros(1);
    f_t=0;
    gamma_t=0.1; 
    SV(:,N)=x_t;         %putting the first sample in the SV list 
    Y(N,1)=y_t;          %Y is the matrix containing labels of the SV's 
    G(N,1)=gamma_t;      %G is the matrix containing the gamma values for each SV
    Index(N,1)=index;    %Index is the matrix containing indexes of the SV's in the sequence
    if y_t==1
        pos(N,1)=1;    %pos is the matrix containing the count for the number of close neigbors with +1 label
        neg(N,1)=0;
    else 
        neg(N,1)=1;    %neg is the matrix containing the count for the number of close neigbors with -1 label
        pos(N,1)=0;
    end

end
hat_y_t=y_t;

if index>1
    for l=1:size(SV,2)
        K(l,1) = kernel(ker,SV(:,l),x_t,p1,p2);          
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
   if (l_t > 0) %if margin is less than one
%        f_t=f_t+(G(N-1,1)*Y(N-1,1)*K(N-1,1))  %updating the model by adding the new sample to the expansion
       SV(:,N)=x_t;         %adding the sample to the SV set 
       gamma_t = min(C,l_t/s_t); % PA-I
%       gamma_t = l_t/(s_t+1/(2*C)); % PA-II
%       gamma_t=1;  %as in original RBP
       Y(N,1)=y_t;          %Y is the matrix containing labels of the SV's 
       G(N,1)=gamma_t;      %G is the matrix containing the gamma values for each SV
       Index(N,1)=index;    %Index is the matrix containing indexes of the SV's in the sequence
       if y_t==1
          pos(N,1)=1;    %pos is the matrix containing the count for the number of close neigbors with +1 label
          neg(N,1)=0;
       else 
          neg(N,1)=1;    %neg is the matrix containing the count for the number of close neigbors with -1 label
          pos(N,1)=0;
       end
   else  %if margin is greater than or equal to 1 
       if y_t==1
           [pos,neg]=updatecount(SV,x_t,K,1,0,pos,neg,ker,p1,p2);
       else 
           [pos,neg]=updatecount(SV,x_t,K,0,1,pos,neg,ker,p1,p2);
       end 
   end 
end
tic   
% removing one SV if size of support set exceeds the budget size B
N=size(SV,2);
if N>B
%     [SV,Y,G,pos,neg,Index,K2]=updateSV(SV,G,Y,pos,neg,ker,p1,p2,Index); %calling updateSV function to perform the removal
   [SV,Y,G,pos,neg,Index,K2]=updateSV(SV,G,Y,pos,neg,ker,p1,p2,Index,K2); %calling updateSV function to perform the removal
end

t(index,1)=toc;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% function [SV2,Y2,G2,pos2,neg2,Index]=updateSV(SV,G,Y,pos,neg,ker,p1,p2,Index);
% %This function removes one SV when |SV|>B
% %INPUTS:
% %       f_t:current model
% %       SV: current SV set
% %       K:  Kernel matrix
% %       G:  matrix containing gamma values
% %       Y:  matrix containing labels of SV's
% 
% %OUTPUTS:
% %       SV2,Y2,G2,pos2,neg2: updated sets
% 
% SV2=SV;
% Y2=Y;
% G2=G;
% pos2=pos;
% neg2=neg;
% N=size(SV,2);
% for i=1:N
%     SV=SV2;
%     Y=Y2;
%     G=G2;
%     pos=pos2;
%     neg=neg2;
%     SV(:,i)=[];
%     G(i,:)=[];
%     Y(i,:)=[];
%     pos(i,:)=[];
%     neg(i,:)=[];
%     for l=1:size(SV,2)
%         for m=1:size(SV,2)
%             K(m,l)=kernel(ker,SV(:,l),SV(:,m),p1,p2); 
%             F(m,l)=G(m,1)*Y(m,1)*K(m,l);
%         end
%         f_t(l,1)=sum(F(:,l))-F(l,l);   %multiplying each kernel with proper label and gamma
%         l_pos= max(0,1-(1*f_t(l,1)));  %Hing loss if the actual label is +1
%         l_neg= max(0,1-(-1*f_t(l,1))); %Hing loss if the actual label is +1
% 
%         pos_posterior= pos(l,1)/(pos(l,1)+neg(l,1));
%         neg_posterior= neg(l,1)/(pos(l,1)+neg(l,1));
%         loss(l,1)=(pos_posterior*l_pos)+(neg_posterior*l_neg);
%     end
%     
%     LOSS(i,1)=(1/N-1)*sum(loss);
%     
% end
% [minimum j]=min(LOSS);
% %removing the j-th SV from the set and updating other SV's counts accordingly. 
% S=SV2(:,j);
% pos=pos2(j,1);
% neg=neg2(j,1);
% SV2(:,j)=[];
% Y2(j,:)=[];
% G2(j,:)=[];
% pos2(j,:)=[];
% neg2(j,:)=[];
% Index(j,:)=[];
%  
% [pos2,neg2]=updatecount(SV2,S,pos,neg,pos2,neg2,ker,p1,p2);
%  
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [SV,Y,G,pos,neg,Index,K2]=updateSV(SV,G,Y,pos,neg,ker,p1,p2,Index);
% %This function removes one SV when |SV|>B
% %INPUTS:
% %       f_t:current model
% %       SV: current SV set
% %       K:  Kernel matrix
% %       G:  matrix containing gamma values
% %       Y:  matrix containing labels of SV's
% 
% %OUTPUTS:
% %       SV2,Y2,G2,pos2,neg2: updated sets
% 
% % SV2=SV;
% % Y2=Y;
% % G2=G;
% % pos2=pos;
% % neg2=neg;
% N=size(SV,2);
% 
% % for p=1:size(SV,2)
% %     for m=1:size(SV,2)
% %         K(m,p)=kernel(ker,SV(:,p),SV(:,m),p1,p2); 
% %         F(m,p)=G(m,1)*Y(m,1)*K(m,p);
% %     end
% % end
% 
% %Calculating just half of the kernel matrix since matrix is symmetric. The
% %other half is duplicated.
% for p=1:size(SV,2)
%     for m=1:size(SV,2)
%         if (p+m)<=size(SV,2)  
%             K(p+m,p)=kernel(ker,SV(:,p),SV(:,p+m),p1,p2);
%         end
%     end
% end
% K(:,end+1)=0;
% 
% for o=1:size(SV,2)
%     for z=1:size(SV,2)
%         if K(z,o)==0
%             K(z,o)=K(o,z);
%         end
%     end
% end
% 
% %Multiplying each kernel value with the corresponding gamma and label of
% %each SV.
% K2=K;
% G2=repmat(G,1,size(K,2));
% Y2=repmat(Y,1,size(K,2));
% F=G2.*Y2.*K2;
% 
% for i=1:size(SV,2)
%     F2=F;
%     pos2=pos;
%     neg2=neg;
%     F2(:,i)=[];
%     F2(i,:)=[];
%     pos2(i,:)=[];
%     neg2(i,:)=[];
%     for l=1:size(F2,2)
%         f_t(l,1)=sum(F2(:,l))-F2(l,l);   %multiplying each kernel with proper label and gamma
%         l_pos= max(0,1-(1*f_t(l,1)));  %Hing loss if the actual label is +1
%         l_neg= max(0,1-(-1*f_t(l,1))); %Hing loss if the actual label is +1
%         pos_posterior= pos2(l,1)/(pos2(l,1)+neg2(l,1));
%         neg_posterior= neg2(l,1)/(pos2(l,1)+neg2(l,1));
%         loss(l,1)=(pos_posterior*l_pos)+(neg_posterior*l_neg);
%     end
%     LOSS(i,1)=(1/N-1)*sum(loss);
% end
%     
% [minimum j]=min(LOSS);
% %removing the j-th SV from the set and updating other SV's counts accordingly. 
% S=SV(:,j);
% pos_count=pos(j,1);
% neg_count=neg(j,1);
% SV(:,j)=[];
% Y(j,:)=[];
% G(j,:)=[];
% pos(j,:)=[];
% neg(j,:)=[];
% Index(j,:)=[];
% 
% k=K2(:,j);
% k(j,:)=[]; %k is the kernel values of the SV which is removed with the rest of SV's which is used in the updatecount function
% 
% [pos,neg]=updatecount(SV,S,k,pos_count,neg_count,pos,neg,ker,p1,p2);
% 
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [SV,Y,G,pos,neg,Index,K2]=updateSV(SV,G,Y,pos,neg,ker,p1,p2,Index,K2);
%This function removes one SV when |SV|>B
%INPUTS:
%       f_t:current model
%       SV: current SV set
%       G:  matrix containing gamma values
%       Y:  matrix containing labels of SV's
%       pos: matrix with number of positive counts for SV's 
%       neg: matrix with number of negative counts for SV's 
%       ker,p1,p1: kernel type and its parameters
%       K2: kernel matrix for SV set (BxB symmetric matrix)

%OUTPUTS:
%       SV,Y,G,pos,neg,Index,K2: updated sets

N=size(SV,2);

% for p=1:size(SV,2)
%     for m=1:size(SV,2)
%         K(m,p)=kernel(ker,SV(:,p),SV(:,m),p1,p2); 
%         F(m,p)=G(m,1)*Y(m,1)*K(m,p);
%     end
% end

%Calculating just half of the kernel matrix since matrix is symmetric. The
%other half is duplicated.
if size(K2,2)==0 % if this is the first call for updateSV function and therefore K2 is not calculated previously, calculate K2 from scrach,O.W K2 just gets updated. 

   for p=1:size(SV,2)
       for m=1:size(SV,2)
           if (p+m)<=size(SV,2)  
               K2(p+m,p)=kernel(ker,SV(:,p),SV(:,p+m),p1,p2);
           end
       end
   end
   K2(:,end+1)=0;

   for o=1:size(SV,2)
       for z=1:size(SV,2)
           if K2(z,o)==0
               K2(z,o)=K2(o,z);
           end
       end
   end
   
else
    
    for l=1:size(SV,2)-1 %this loop calculates the kernel of the new added SV with the rest of SV's to update kernel matrix
        new_K_row(1,l)=kernel(ker,SV(:,l),SV(:,end),p1,p2);
    end
    new_K_col=new_K_row';
    new_K_col(end+1,1)=0;
%     size_K2=size(K2)
%     size_new_K_row=size(new_K_row)
    K2=cat(1,K2,new_K_row);
    K2=cat(2,K2,new_K_col);
    
end

%Multiplying each kernel value with the corresponding gamma and label of
%each SV.
G2=repmat(G,1,size(K2,2));
Y2=repmat(Y,1,size(K2,2));
F=G2.*Y2.*K2;

for i=1:size(SV,2)
    F2=F;
    pos2=pos;
    neg2=neg;
    F2(:,i)=[];
    F2(i,:)=[];
    pos2(i,:)=[];
    neg2(i,:)=[];
    for l=1:size(F2,2)
        f_t(l,1)=sum(F2(:,l))-F2(l,l);   %multiplying each kernel with proper label and gamma
        l_pos= max(0,1-(1*f_t(l,1)));  %Hing loss if the actual label is +1
        l_neg= max(0,1-(-1*f_t(l,1))); %Hing loss if the actual label is +1
        pos_posterior= pos2(l,1)/(pos2(l,1)+neg2(l,1));
        neg_posterior= neg2(l,1)/(pos2(l,1)+neg2(l,1));
        loss(l,1)=(pos_posterior*l_pos)+(neg_posterior*l_neg);
    end
    LOSS(i,1)=(1/N-1)*sum(loss);
end
    
[minimum j]=min(LOSS);
%removing the j-th SV from the set and updating other SV's counts accordingly. 
S=SV(:,j);
pos_count=pos(j,1);
neg_count=neg(j,1);
SV(:,j)=[];
Y(j,:)=[];
G(j,:)=[];
pos(j,:)=[];
neg(j,:)=[];
Index(j,:)=[];

k=K2(:,j);
k(j,:)=[]; %k is the kernel values of the SV which is removed with the rest of SV's which is used in the updatecount function

K2(:,j)=[];
K2(j,:)=[];

[pos,neg]=updatecount(SV,S,k,pos_count,neg_count,pos,neg,ker,p1,p2);
 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [pos,neg]=updatecount(SV,x,K,pos_count,neg_count,pos,neg,ker,p1,p2)
% This function updates the number of positive and negative sample labels in the close neighborhood  
% OUTPUTS:   
%       pos: pos matrix with updated number of counts for positive labels for SV(:,I) 
%       neg: neg matrix with updated number of counts for negative labels for SV(:,I)

%Euclidean distance in original space:
k=1; %k nearest points in the SV set to x 
[D,I] = pdist2(SV',x','euclidean','Smallest',k); %I is the index of the SV (in the SV set) with the smallest euclidean distance to x

% %distance in feature space:
% for i=1:size(SV,2)
% %     K(i,1)=kernel(ker,SV(:,i),x,p1,p2);
%     dist(i,1)=sqrt(2*(1-K(i,1)));
% end
% [D,I]=min(dist);
    
K=kernel(ker,SV(:,I),x,p1,p2); %This is in the case of Euclidean
% distance
% K=K(I,1); %This is in the case of distance in feature space

pos(I,1)= pos(I,1)+pos_count*K;
neg(I,1)= neg(I,1)+neg_count*K;

% pos(I,1)= pos(I,1)+pos_count;
% neg(I,1)= neg(I,1)+neg_count;
end

