function initZ = initializeChain(X, y, noiseBeta, p0, invVs)
% Sets up the initial value of Z for the Gibbs sampler iterations
% X --> Regressor (or feature) matrix
% y --> Column vector of measurements
% noiseBeta --> Measurement noise precision (1/sigma_0^2)
% p0 --> fraction of components of Z that are different from 0
% invVs --> Inverse of variance of the slab

d   = size(X,2);
r  = floor(p0*d);

selectedCols = [];

for i = 1:r
    %Computes the current residual
    e = computeResidual(selectedCols, X, y, noiseBeta, invVs); 

    correlations = zeros(1,d);
    for j = 1:d
         correlations(j) = abs(corr(e, X(:,j)));
    end

    [~,newFeatureIdx] = max(correlations);
    selectedCols = [selectedCols, newFeatureIdx];
end

initZ = zeros(1,d);
initZ(selectedCols) = 1;

end

function e = computeResidual(selectedCols, X, y, noiseBeta, invVs)
    rr  = length(selectedCols);
    if(rr~=0)
        Xactive  = X(:,selectedCols);
        InvSigma = invVs*eye(rr) + noiseBeta*(Xactive'*Xactive);
        meanW    = noiseBeta*(InvSigma\(Xactive'*y));
        e        = y - Xactive*meanW;
    else
        e = y;
    end
end




