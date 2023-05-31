function out = BVS_basad2(X, y, nSamples, nburn, nchains, verbosity)

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

W0 = zeros(1,p+1);
Z0 = zeros(1,p+1);

p0 = 0.1; 
v0 = 1/n;
v1 = 100*v0;
tau2 = 1;

if(nargin<6)
    verbosity = false;
end

results = cell(nchains,1);
for i = 1:nchains
    sig2_i      = 1/(noiseBeta*abs(10*randn));
    tau2_i      = tau2*abs(10*randn);
    p0_i        = p0;    
    results{i}  = BASAD_Gibbs2(X, y, nSamples, W0, Z0,...
                  sig2_i, tau2_i, p0_i, v0, v1, verbosity);
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

modelIdx = find(Zmean > 0.5);
modelIdx = setdiff(modelIdx,1);

% disp(['Wmean: ', num2str(Wmean(modelIdx))])

out.DS = results;
out.ZZ = ZZ;
out.WW = WW;
out.sig2 = mean(sig2);
out.modelIdx = modelIdx-1;
out.Zmed = Zmean(modelIdx);
out.Wsel = Wmean(modelIdx);
out.Wcov = Wcov(modelIdx, modelIdx);
out.Rw   = Rw;
end


function DS = BASAD_Gibbs2(X, y, nSamples, W0, Z0, sig2, tau2, p0, v0, v1, Verbosity)
% X     : Design matrix
% y     : Observations
% nSamples: Total number of chain samples (including burn-in)
% W0    : Initial weight or coefficient vector
% Z0    : Initial latent variable vector

% v1    : fixed multiplier of slab variance
% v0    : fixed multiplier of spike variance
% tau2  : varying scale for the spike and the slab
% sig2  : noise variance
% p0    : inclusion probability

% Deterministic prior parameters
% (a) IG dist (for noise variance sigma^2)
% Values chosen leads to closely non-informative prior IG(0,0)
alpha0      = 1e-4;
beta0       = 1e-4;
% (b) IG dist (for slab scale tau^2)
% degree of freedom nu = 1, s^2 = 1
% Leads to Cauchy(0,1) distribution of the slab
nu = 1; s2 = 1;
alphad0     = nu/2;
betad0      = nu*s2/2;
% (c) Beta dist (for inclusion prob p0)
% Beta(0.1,1) leads to informative prior causing more sparse solutions
a0          = 0.1;
b0          = 1;

[n,d]       = size(X);

% Perform some initializations
samplesSig2 = zeros(nSamples, 1);   % Sigma^2 (noise variance)
samplesP    = zeros(nSamples, 1);   % P0 (inclusion probability)
samplesTau2 = zeros(nSamples, d);   % Tau^2 (predictor-specific slab scale variance)
samplesZ    = zeros(nSamples, d);
samplesW    = zeros(nSamples, d);

samplesSig2(1)  = sig2;
samplesP(1)     = p0;
samplesTau2(1,:)= tau2*ones(1,d);     
samplesZ(1,:)   = Z0;
samplesW(1,:)   = W0;

GG      = X'*X;
tmu     = X'*y;
eyed    = eye(d);
zerosd  = zeros(1,d);

% Start sampling from the conditional distributions
for i = 2:nSamples
    
    prevZ           = samplesZ(i-1,:);
    prevT           = samplesTau2(i-1,:);
    prevP           = samplesP(i-1);
    prevsig2        = samplesSig2(i-1);
    
    % Update the weights W
    D               = diag( 1./(v1 * (prevZ.*prevT) + v0 * ((1-prevZ).*prevT)) );
    invSigmaz       = GG + D;
    invSigmaz       = 0.5*(invSigmaz + invSigmaz');
    Sigmaz          = invSigmaz\eyed;
    muW             = Sigmaz*tmu;
    newW            = muW' + sqrt(prevsig2)*randn(1,d)*cholcov(Sigmaz); 
    samplesW(i,:)   = newW;
    
    % Update the latent variables Z
    term0           = (1-prevP) * mydnorm(newW, zerosd, prevsig2*v0*prevT);
    term1           = prevP * mydnorm(newW, zerosd, prevsig2*v1*prevT);
    pratio          = term1./(term0 + term1);
    newZ            = rand(1,d) < pratio;
    newZ(1)         = 1;
    samplesZ(i,:)   = newZ;
        
    % Update sigma^2
    e               = y - X*newW(:);
    alpha           = alpha0 + 0.5*n + 0.5*(d-1);
    beta            = beta0 + 0.5*(e'*e) + 0.5*newW * D * newW';
    newsig2         = 1/gamrnd(alpha, 1/beta); 
    samplesSig2(i)  = newsig2;
    
    % Update tau^2 (each coefficient has a separate tau)
    T1              = v1*newZ + v0*(1-newZ);
    alphad          = alphad0 + 0.5;
    betad           = betad0 + 0.5*(newW.^2) ./ (2*newsig2*T1);
%     newtau2         = 1./gamrnd(alphad, 1./betad);
%     samplesTau2(i,:)= newtau2;
    for jj = 1:d
        newtau2     = 1/gamrnd(alphad, 1/betad(jj));
        samplesTau2(i,jj)= newtau2;
    end
    
    % Update the P from beta distribution
    sz              = sum(newZ);
    samplesP(i)     = betarnd(a0 + sz, b0 + d - sz);
    
    if(Verbosity)
        if(mod(i,100)==0)
            fprintf('     Iteration %d\n',i);
        end
    end
end


DS.samplesW    = samplesW;
DS.samplesZ    = samplesZ;
DS.samplesP    = samplesP;
DS.samplesSig2 = samplesSig2;
DS.samplesTau2 = samplesTau2;

DS.prior.sig2  = [alpha0, beta0];
DS.prior.tau2  = [alphad0, betad0];
DS.prior.p0    = [a0, b0];

end

function pdfxx = mydnorm(W, mu, sigsq)

pdfxx = (1./sqrt(2*pi*sigsq)) .* exp(-((W-mu).^2)./(2*sigsq));

end
