function[dQdI2]=diffFeaturewith_I2(I3,I1,I2,epsilon,N,M,Q)
J=sqrt(I3);
K1 = (J.^(-2/3)).*I1-3.0*ones(size(I2));  %  k1=(I1(bar)-3)
K2 = (J.^(-4/3)).*(I1 + I3 - ones(size(I2)))-3.0*ones(size(I2)); % k2=(I2(bar)-3) 
I11=I1;
I22=I2+epsilon*ones(size(I2));
I33=I3;
JJ=sqrt(I33);
K11 = (JJ.^(-2/3)).*I11-3.0*ones(size(I22));
K22 = (JJ.^(-4/3)).*(I11 + I33 - ones(size(I33)))-3.0*ones(size(I22));
% Calculate the number of features.
numFeatures = 0;
if N~=0
    for n=1:N
        numFeatures=numFeatures+n+1;
    end
end
if M~=0
    numFeatures=numFeatures+M;
end
QQ = zeros(length(I2),numFeatures);
k=0;

%Polynomial terms (dependent on K1 and K2).
for j=1:N
    for ii=1:j+1
        i=ii-1;
        k=k+1; 
        QQ(:,k)= (K11.^(i)).* K22.^(j-i);
    end
end
%Volumetric terms (dependent on J):
for m=1:M
    k=k+1; 
    QQ(:,k) = (JJ-ones(size(I11))).^(2*m);
end
    k=k+1;
    QQ(:,k) = log((K22+3.0)/3.0);
    dQdI2=(1/epsilon).*(QQ-Q);
end