function[Q_g] = featurematrix(J_g,I1_bar_g,I2_bar_g,N,M,NF)
    K1_g = I1_bar_g - 3;
    K2_g = I2_bar_g - 3;
    %Additional Gent-Thomas feature.
    %if considerGentThomas
    %numFeatures += 1
    %Calculate the features.
    Q_g = zeros(NF,1);
    k=0;
    %Polynomial terms (dependent on K1 and K2).
    for j=1:N
        for ii=1:j+1
            i=ii-1;
            k=k+1; 
            Q_g(k,1) = (K1_g^(i))*K2_g^(j-i);
        end
    end
    %Volumetric terms (dependent on J):
    for m=1:M
        k=k+1; 
        Q_g(k,1) = (J_g-1)^(2*m);
    %Additional Gent-Thomas feature.
    %if considerGentThomas:
        %i+=1; x[:,i:(i+1)] = torch.log((K2+3.0)/3.0)
    %end
    end
    k=k+1;
    Q_g(k,1) = log((K2_g+3.0)/3.0);
end