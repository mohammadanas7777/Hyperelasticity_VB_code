function out = BVS_Walli_G(X, y, nSamples, nburn, nchains, verbosity)
% Dirac spike and Zellner's g-slab prior
% with variance of the slab dependent upon noise variance sigma^2
% This is marginal likelihood implementation of Walli (2010)

if(isempty(X) || isempty(y))
    error('X and or y is missing');
end
if(size(X,1) ~= size(y,1))
    error('Number of observations do not match');
end

X = normalize(X);   % Standardise the columns (mean 0 and sd 1)
X = [ones(size(X,1),1), X];    % Add a constant vector of ones
y = y - mean(y);    % Detrend the measurements

p = size(X,2)-1;
n = size(X,1);

W0 = zeros(1,p+1);
for i = 1:p+1
    W0(i) = X(:,i)\y;
end
if(p>n)
    m = round(p - n/2);
else
    m = round(p/2);
end

% Get an estimate of sig2
sortW0  = sort(abs(W0));
ind     = find(abs(W0) > sortW0(m+1));
ind2    = unique(union(1, ind));
tmpXX   = X(:,ind2);
tmpcoef = tmpXX \ y;
sig2    = sum((y - tmpXX*tmpcoef).^2)/(n-length(ind2));

noiseBeta = 1/sig2; % noise precision (inverse of variance)

p0 = 0.1;   % Initial inclusion probability
vs = 10;    % Slab variance 

invA0 = X'*X/(n-1);

Z0 = initializeChain(X, y, p0, 1/vs, invA0);
Z0(1) = 1;  % Always include the intercept term
W0 = SampleW(Z0, X, y, noiseBeta, 1/vs, invA0);

% load dataI.mat

results = cell(nchains,1);
for i = 1:nchains
    noiseBeta_i = noiseBeta*abs(10*randn);
    vs_i        = vs*abs(10*randn);
    p0_i        = p0;    
    results{i}  = DISSP_Zellgslab_Gibbs(X, y, nSamples, W0, Z0,...
                  noiseBeta_i, vs_i, p0_i, verbosity);
end

dd = p+1;
% Assembling the results from chains
nsamp = nSamples - nburn;
ZZ  = nan(nchains*nsamp, dd);
WW  = nan(nchains*nsamp, dd);
sig2= nan(nchains*nsamp, 1);
Wstr= nan(nsamp, dd, nchains);

for i = 1:nchains
    Wstr(:,:,i)  = results{i}.samplesW(nburn+1:end,:);
    
    arange       = ((i-1)*nsamp + 1) : i*nsamp;
    ZZ(arange,:) = results{i}.samplesZ(nburn+1:end,:);
    WW(arange,:) = Wstr(:,:,i);
    sig2(arange,:)= results{i}.samplesSig2(nburn+1:end,:);
end

% Multivariate PSRF (gelman-rubin stat)
Rw = mpsrf_jitter(Wstr);

Zmean = mean(ZZ);
Wmean = mean(WW);
Wcov  = cov(WW);

modelIdx = find(Zmean > 0.5);   % Median prob model
modelIdx = setdiff(modelIdx,1);

out.DS = results;
out.ZZ = ZZ;
out.WW = WW;
out.sig2 = mean(sig2);
out.modelIdx = modelIdx-1;
out.Zmed = Zmean(modelIdx);
out.Wsel = Wmean(modelIdx);
out.Wcov = Wcov(modelIdx, modelIdx);
out.Rw = Rw;

end


%% FUNCTION FOR GIBBS SAMPLING with a common slab variance scale
function DS = DISSP_Zellgslab_Gibbs(X, y, nSamples, W0, Z0, noiseBeta, vs, p0, verbosity)
% X     : Design matrix
% y     : Observations
% nSamples: Total number of chain samples (including burn-in)
% W0    : Initial weight or coefficient vector
% Z0    : Initial latent variable vector

% sig2  : noise variance
% vs    : varying scale for the slab (could be single or a vector
% p0    : inclusion probability     

% Deterministic prior parameters
% (a) IG dist (for noise variance sigma^2) 
% Values chosen leads to closely non-informative prior IG(0,0)
alpha0      = 1e-4;
beta0       = 1e-4;
% (b) IG dist (for slab scale vs)
% degree of freedom nu = 1, s^2 = 1
% Leads to Cauchy(0,1) distribution of the slab
nu = 1; s2 = 1;
alphad0     = nu/2;
betad0      = nu*s2/2;
% (c) Beta dist (for inclusion prob p0)
% Beta(0.1,1) leads to informative prior causing more sparse solutions
a0          = 0.1;
b0          = 1;
% The ration a0/b0 means roughly 1 in 10 models would be selected

[n,d]       = size(X);

% Perform some initializations
samplesSig2 = zeros(nSamples, 1);   % Sigma^2 (noise variance)
samplesP    = zeros(nSamples, 1);   % P0 (inclusion probability)
samplesVs   = zeros(nSamples, 1);   % vs (slab scale variance)
samplesZ    = zeros(nSamples, d);
samplesW    = zeros(nSamples, d);

samplesSig2(1)  = 1/noiseBeta;
samplesP(1)     = p0;
samplesVs(1,:)  = vs;     
samplesZ(1,:)   = Z0;
samplesW(1,:)   = W0;

invVs           = 1/vs;
invA0           = X'*X/(n-1);

% Start sampling from the conditional distributions
for i = 2:nSamples
    
    prevZ           = samplesZ(i-1,:);
    prevW           = samplesW(i-1,:);
    
    % Sample P from beta distribution
    sz              = sum(prevZ);
    newP            = betarnd(a0 + sz, b0 + d - sz);
    samplesP(i)     = newP;
    
    % Sample noiseBeta from a Gamma distribution
    eyer            = eye(sz);
    idxz            = find(prevZ == 1);
    Xz              = X(:,idxz);
%     g               = (n-1)/invVs;
    invA0z          = invA0(idxz, idxz);
    invSigmaz       = (Xz'*Xz) + invVs*invA0z;
    invSigmaz       = 0.5*(invSigmaz + invSigmaz');
    Adelta          = invSigmaz\eyer;
    Xty             = Xz'*y;
    Sn              = y'*y - Xty'*Adelta*Xty; % y'*y - adelta'*(Adelta\adelta)
    
    noiseBeta       = gamrnd( alpha0 + 0.5*n, 1/(beta0 + 0.5*Sn) ); 
    samplesSig2(i)  = 1/noiseBeta;
    
    % Sample invVs (inverse of vs) from a inverse Gamma distribution
    Wz              = prevW(idxz);
    alphad          = alphad0 + 0.5*sz;
    betad           = betad0 + 0.5*noiseBeta*(Wz*invA0z*Wz');
    invVs           = gamrnd(alphad, 1/betad);
    samplesVs(i)    = 1/invVs;
    
    % Update the weights Z (sample each z_j in random order)
    order   = setdiff(randperm(d), 1, 'stable');
    newZ    = prevZ;
    for j = order
        z0          = newZ;
        
        z0(j)       = 0;
        log_ml_yz0_vs = marlike(X, y, z0, invVs, alpha0, beta0, invA0);
        z0(j)       = 1;
        log_ml_yz1_vs = marlike(X, y, z0, invVs, alpha0, beta0, invA0);
        
        % Sample z_j from a Bernoulli distribution
        pratio = newP/(newP + exp(log_ml_yz0_vs - log_ml_yz1_vs)*(1-newP) );
%         pratio = exp(f1)/(exp(f0) + exp(f1))
    
        ZZ = binornd( 1, pratio );
        
        newZ(j) = ZZ;
        
    end
%     newZ(1) = 1;    % always included the constant coefficient 1
    samplesZ(i,:) = newZ;
    
    % Sample weights from multivariate Gaussian
    newW            = SampleW(newZ, X, y, noiseBeta, invVs, invA0);
    samplesW(i,:)   = newW;
    
    if(verbosity)
        if(mod(i,500)==0)
            fprintf('     Iteration %d\n',i);
        end
    end
end

DS.samplesW    = samplesW;
DS.samplesZ    = samplesZ;
DS.samplesP    = samplesP;
DS.samplesSig2 = samplesSig2;
DS.samplesVs   = samplesVs;

DS.prior.sig2  = [alpha0, beta0];
DS.prior.vs    = [alphad0, betad0];
DS.prior.p0    = [a0, b0];

end

%% MARGINAL LOG-LIKELIHOOD for sampling z (following Walli-Malsiner 2010)
function logML = marlike(X, y, Z, invVs, alpha0, beta0, invA0) 

N       = size(X,1);    % number of observations

sz      = sum(Z);       % number of active z-variables
eyesz   = eye(sz);
indx    = find(Z==1);   % Finding indices of Z which are non-zero
XXz     = X(:,indx);
invA0d  = invA0(indx, indx);
% vsn     = (N-1)/invVs;      % g = n * vs (vsn can be thought as modified vs)
% invAz   = (1+vsn)/vsn * A0z;
invAz   = (XXz'*XXz) + invVs*invA0d;
invAz   = 0.5*(invAz + invAz');
Az      = invAz \ eyesz;

XXty    = XXz'*y;
SN      = 0.5*(y'*y - XXty'*Az*XXty);

logML   = -0.5*N*log(2*pi) + 0.5*logdet(Az) + 0.5*logdet(invA0d) + alpha0*log(beta0)...
          -(alpha0+0.5*N)*log(beta0+SN) + gammaln(alpha0+0.5*N) - gammaln(alpha0)...
          + 0.5*sz*log(invVs);
    
end

function out = logdet(A)
    out  = 2*sum(log(diag(chol(A))));
end
        
%% SAMPLE W functions
function [newW, muW] = SampleW(Z, X, y, noiseBeta, invVs, invA0)
% Samples a new vector w
% Z --> Vector of activations
% X --> Feature Matrix
% y --> Measurement vector
% noiseBeta --> Noise precision (1/sigma_0^2)
% invVs --> inverse of slab variance

d = size(Z,2);      % Total number of features is d
% ny = length(y);     % Total number of observations

newW = zeros(1,d);  % Initializing the new sample of W with zeros
muW  = zeros(d,1);

idx = find(Z==1);       % Finding indices of Z which are non-zero

if(~isempty(idx))
    r = length(idx);    % Total number of active (non-zero) z components
    eyer = eye(r);
    Xz              = X(:,idx);
%     gg              = (ny-1)/invVs;     % g = n * vs
%     invSigmaz       = (gg+1)/gg * (Xz'*Xz);
    invA0zz         = invA0(idx, idx);
    invSigmaz       = (Xz'*Xz) + invVs*invA0zz;
    invSigmaz       = 0.5*(invSigmaz + invSigmaz');
    Sigmaz          = invSigmaz\eyer;       % A_delta
    
    meanW           = Sigmaz*Xz'*y;         % a_delta
    muW(idx,1)      = meanW;
    newW(:,idx)     = meanW' + randn(1,r)*cholcov(noiseBeta\Sigmaz);
end

end

%% FUNCTION FOR INITIALIZING VALUES OF Z
function initZ = initializeChain(X, y, p0, invVs, invA0)
% Sets up the initial value of Z for the Gibbs sampler iterations
% X --> Regressor (or feature) matrix
% y --> Column vector of measurements
% p0 --> fraction of components of Z that are different from 0
% vs --> variance of the slab

d   = size(X,2);
r  = floor(p0*d);

selectedCols = [];

for i = 1:r
    %Computes the current residual
    e = computeResidual(selectedCols, X, y, invVs, invA0); 

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

function e = computeResidual(selectedCols, X, y, invVs, invA0)
    rr  = length(selectedCols);
%     nn  = length(y);
%     g   = (nn-1)/invVs;     % g = n * vs
    if(rr~=0)
        Xactive  = X(:,selectedCols);
        invA0z   = invA0(selectedCols, selectedCols);
        InvSigma = (Xactive'*Xactive) + invVs*invA0z;  % Adelta = g/(g+1) * (X'*X)^{-1}
        meanW    = InvSigma\(Xactive'*y);
        e        = y - Xactive*meanW;
    else
        e = y;
    end
end




