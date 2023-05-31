%Compute the features dependent on the right Cauchy-Green strain invariants.
%Note that the features only depend on I1 and I3.
function[Q]=computeFeatures(J,I1,I3,N,M,NF)
    %Generalized Mooney-Rivlin.
    %The Gent-Thomas model cannot be represented by the generalized
    %Mooney-Rivlin model. An additional feature has to be added.
    
    K1 = (J.^(-2/3)).*I1 - 3;
    K2 = (J.^(-4/3)).*(I1 + I3 - 1) - 3;
    %Additional Gent-Thomas feature.
    %if considerGentThomas
    %numFeatures += 1
    %Calculate the features.
    Q = zeros(length(I1),NF);
    k=0;
    %Polynomial terms (dependent on K1 and K2).
    for j=1:N
        for ii=1:j+1
            i=ii-1;
            k=k+1; 
%             Q(:,k) = K1.^(j-i) .* K2.^i;
            Q(:,k) = (K1.^(i)).* K2.^(j-i);
        end
    end
    %Volumetric terms (dependent on J):
    for m=1:M
        k=k+1; 
        Q(:,k) = (J-ones(size(I1))).^(2*m);
    %Additional Gent-Thomas feature.
    %if considerGentThomas:
        %i+=1; x[:,i:(i+1)] = torch.log((K2+3.0)/3.0)
    %end
    end
    k=k+1;
    Q(:,k) = log((K2+3.0)/3.0);
end
