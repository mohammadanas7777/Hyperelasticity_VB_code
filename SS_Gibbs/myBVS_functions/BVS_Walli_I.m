function out = BVS_Walli_I(X, y, nSamples, nburn, nchains, verbosity)
% Dirac spike and independent slab prior
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
vs = 10;     % Slab variance 

Z0 = initializeChain(X, y, p0, 1/vs);
Z0(1) = 1;  % Always include the intercept term
W0 = SampleW(Z0, X, y, noiseBeta, 1/vs);

% load dataI.mat

results = cell(nchains,1);
for i = 1:nchains
    noiseBeta_i = noiseBeta*abs(10*randn);
    vs_i        = vs*abs(10*randn);
    p0_i        = p0;    
    results{i}  = DISSP_indslab_Gibbs(X, y, nSamples, W0, Z0,...
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
function DS = DISSP_indslab_Gibbs(X, y, nSamples, W0, Z0, noiseBeta, vs, p0, verbosity)
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
alpha0      = 0.1;
beta0       = 0.1;
% (b) IG dist (for slab scale vs)
% degree of freedom nu = 1, s^2 = 1
% Leads to Cauchy(0,1) distribution of the slab
nu = 1; s2 = 1;
alphad0     = nu/2;
betad0      = nu*s2/2;
% (c) Beta dist (for inclusion prob p0)
% Beta(0.1,1) leads to informative prior causing more sparse solutions
a0          = 0.1;
b0          = 5;
% The ration a0/b0 means roughly 1 in 10 models would be selected

[n,d]       = size(X);

% Perform some initializations
samplesSig2 = zeros(nSamples, 1);   % Sigma^2 (noise variance)
samplesP    = zeros(nSamples, 1);   % P0 (inclusion probability)
samplesVs   = zeros(nSamples, 1);   % vs (slab scale variance)
samplesZ    = zeros(nSamples, d);
samplesW    = zeros(nSamples, d);

samplesSig2(1)  = (1/noiseBeta);
samplesP(1)     = p0;
samplesVs(1,:)  = vs;    
samplesZ(1,:)   = Z0;
samplesW(1,:)   = W0;

invVs           = 1/vs;
A0              = eye(d);
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
    Xz              = X(:,prevZ==1);
    invSigmaz       = invVs*eyer +  Xz'*Xz;
    invSigmaz       = 0.5*(invSigmaz + invSigmaz');
    Adelta          = invSigmaz\eyer;
    Xty             = Xz'*y;
    Sn              = y'*y - Xty'*Adelta*Xty; % y'*y - adelta'*(Adelta\adelta)
    noiseBeta       = gamrnd( alpha0 + 0.5*n, 1/(beta0 + 0.5*Sn) );
    samplesSig2(i)  = 1/noiseBeta;
    
    % Sample invVs (inverse of vs) from a inverse Gamma distribution
    alphad          = alphad0 + 0.5*sz;
    betad           = betad0 + 0.5*noiseBeta*sum(prevW.^2);
    invVs           = gamrnd(alphad, 1/betad);
    samplesVs(i)    = 1/invVs;
    
    % Update the weights Z (sample each z_j in random order)
    order   = setdiff(randperm(d), 1, 'stable');
    newZ    = prevZ;
    for j = order
        z0          = newZ;        
        z0(j)       = 0;
        log_ml_yz0_vs = marlike(X, y, z0, invVs, A0, alpha0, beta0);
        z0(j)       = 1;
        log_ml_yz1_vs = marlike(X, y, z0, invVs, A0, alpha0, beta0);
        
        % Sample z_j from a Bernoulli distribution
        pratio = newP/(newP + exp(log_ml_yz0_vs - log_ml_yz1_vs)*(1-newP) );
%         pratio = exp(f1)/(exp(f0) + exp(f1))
    
        ZZ = binornd( 1, pratio );
        
        newZ(j) = ZZ;
        
    end
%     newZ(1) = 1;    % always included the constant coefficient 1
    samplesZ(i,:) = newZ;
    
    % Sample weights from multivariate Gaussian
    newW            = SampleW(newZ, X, y, noiseBeta, invVs);
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
function logML = marlike(X, y, Z, invVs, A0, alpha0, beta0) 

N       = size(X,1);    % number of observations
Z;
sz      = sum(Z);     % number of active z-variables
eyesz   = eye(sz);
indx    = find(Z==1);   % Finding indices of Z which are non-zero
XXz     = X(:,indx);
A0z     = A0(indx, indx);
invAz   = XXz'*XXz + invVs*(A0z\eyesz);
invAz   = 0.5*(invAz + invAz');
Az      = invAz \ eyesz;

XXty    = XXz'*y;
SN      = 0.5*(y'*y - XXty'*Az*XXty);
logML   = -0.5*N*log(2*pi) - 0.5*logdet(A0z) + alpha0*log(beta0) + 0.5*logdet(Az)...
          -(alpha0+0.5*N)*log(beta0+SN) + gammaln(alpha0+0.5*N) - gammaln(alpha0)...
          + 0.5*sz*log(invVs);
    
end

function out = logdet(A)
    out  = 2*sum(log(diag(chol(A))));
end
        
%% SAMPLE W functions
function [newW, muW] = SampleW(Z, X, y, noiseBeta, invVs)
% Samples a new vector w
% Z --> Vector of activations
% X --> Feature Matrix
% y --> Measurement vector
% noiseBeta --> Noise precision (1/sigma_0^2)
% invVs --> inverse of slab variance

d = size(Z,2);      % Total number of features is d

newW = zeros(1,d);  % Initializing the new sample of W with zeros
muW  = zeros(d,1);

idx = find(Z==1);   % Finding indices of Z which are non-zero

if(~isempty(idx))
    r = length(idx);    % Total number of active (non-zero) z components
    eyer = eye(r);
    Xz              = X(:,idx);
    
    invSigmaz       = invVs*eyer +  Xz'*Xz;
    invSigmaz       = 0.5*(invSigmaz + invSigmaz');
    Sigmaz          = invSigmaz\eyer;       % A_delta
    
    meanW           = Sigmaz*Xz'*y;         % a_delta
    muW(idx,1)      = meanW;
    D = (diag(noiseBeta\Sigmaz));
    R=zeros(1,r);
    for i = 1:r
        R(1,i) = meanW(i)+D(i)*trandn((0-meanW(i))/D(i),inf);
    end
    newW(:,idx) = R(1,:);
%     newW(:,idx)     = meanW' + randn(1,r)*cholcov(noiseBeta\Sigmaz);
end

end

%% FUNCTION FOR INITIALIZING VALUES OF Z
function initZ = initializeChain(X, y, p0, invVs)
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
    e = computeResidual(selectedCols, X, y, invVs); 

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

function e = computeResidual(selectedCols, X, y, invVs)
    rr  = length(selectedCols);
    if(rr~=0)
        Xactive  = X(:,selectedCols);
        InvSigma = invVs*eye(rr) + Xactive'*Xactive;
        meanW    = InvSigma\(Xactive'*y);
        e        = y - Xactive*meanW;
    else
        e = y;
    end
end

%% For sampling theta only from positive part of Normal distribution below functions will be used as in line 279
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x=trandn(l,u)
%% truncated normal generator
% * efficient generator of a vector of length(l)=length(u)
% from the standard multivariate normal distribution,
% truncated over the region [l,u];
% infinite values for 'u' and 'l' are accepted;
% * Remark:
% If you wish to simulate a random variable
% 'Z' from the non-standard Gaussian N(m,s^2)
% conditional on l<Z<u, then first simulate
% X=trandn((l-m)/s,(u-m)/s) and set Z=m+s*X;
l=l(:);u=u(:); % make 'l' and 'u' column vectors
if length(l)~=length(u)
    error('Truncation limits have to be vectors of the same length')
end
x=nan(size(l));
a=.66; % treshold for switching between methods
% threshold can be tuned for maximum speed for each Matlab version
% three cases to consider:
% case 1: a<l<u
I=l>a;
if any(I)
    tl=l(I); tu=u(I); x(I)=ntail(tl,tu);
end
% case 2: l<u<-a
J=u<-a;
if any(J)
    tl=-u(J); tu=-l(J); x(J)=-ntail(tl,tu);
end
% case 3: otherwise use inverse transform or accept-reject
I=~(I|J);
if  any(I)
    tl=l(I); tu=u(I); x(I)=tn(tl,tu);
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function p=lnNpr(a,b)
% computes ln(P(a<Z<b))
% where Z~N(0,1) very accurately for any 'a', 'b'
p=zeros(size(a));
% case b>a>0
I=a>0;
if any(I)
    pa=lnPhi(a(I)); % log of upper tail
    pb=lnPhi(b(I));
    p(I)=pa+log1p(-exp(pb-pa));
end
% case a<b<0
idx=b<0;
if any(idx)
    pa=lnPhi(-a(idx)); % log of lower tail
    pb=lnPhi(-b(idx));
    p(idx)=pb+log1p(-exp(pa-pb));
end
% case a<0<b
I=(~I)&(~idx);
if any(I)
    pa=erfc(-a(I)/sqrt(2))/2; % lower tail
    pb=erfc(b(I)/sqrt(2))/2;  % upper tail
    p(I)=log1p(-pa-pb);
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function p=lnPhi(x)
% computes logarithm of  tail of Z~N(0,1) mitigating
% numerical roundoff errors;
p=-0.5*x.^2-log(2)+reallog(erfcx(x/sqrt(2)));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function  x=trnd(l,u)
% uses acceptance rejection to simulate from truncated normal
x=randn(size(l)); % sample normal
% keep list of rejected
I=find(x<l|x>u); d=length(I);
while d>0 % while there are rejections
    ly=l(I); % find the thresholds of rejected
    uy=u(I);
    y=randn(size(ly));
    idx=y>ly&y<uy; % accepted
    x(I(idx))=y(idx); % store the accepted
    I=I(~idx); % remove accepted from list
    d=length(I); % number of rejected
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x=tn(l,u)
% samples a column vector of length=length(l)=length(u)
% from the standard multivariate normal distribution,
% truncated over the region [l,u], where -a<l<u<a for some
% 'a' and l and u are column vectors;
% uses acceptance rejection and inverse-transform method;
tol=2; % controls switch between methods
% threshold can be tuned for maximum speed for each platform
% case: abs(u-l)>tol, uses accept-reject from randn
I=abs(u-l)>tol; x=l;
if any(I)
    tl=l(I); tu=u(I); x(I)=trnd(tl,tu);
end
% case: abs(u-l)<tol, uses inverse-transform
I=~I;
if any(I)
    tl=l(I); tu=u(I); pl=erfc(tl/sqrt(2))/2; pu=erfc(tu/sqrt(2))/2;
    x(I)=sqrt(2)*erfcinv(2*(pl-(pl-pu).*rand(size(tl))));
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x=ntail(l,u)
% samples a column vector of length=length(l)=length(u)
% from the standard multivariate normal distribution,
% truncated over the region [l,u], where l>0 and
% l and u are column vectors;
% uses acceptance-rejection from Rayleigh distr.
% similar to Marsaglia (1964);
c=l.^2/2; n=length(l); f=expm1(c-u.^2/2);
x=c-reallog(1+rand(n,1).*f); % sample using Rayleigh
% keep list of rejected
I=find(rand(n,1).^2.*x>c); d=length(I);
while d>0 % while there are rejections
    cy=c(I); % find the thresholds of rejected
    y=cy-reallog(1+rand(d,1).*f(I));
    idx=rand(d,1).^2.*y<cy; % accepted
    x(I(idx))=y(idx); % store the accepted
    I=I(~idx); % remove accepted from list
    d=length(I); % number of rejected
end
x=sqrt(2*x); % this Rayleigh transform can be delayed till the end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ L, l, u, perm ] = cholperm( Sig, l, u )
%  Computes permuted lower Cholesky factor L for Sig
%  by permuting integration limit vectors l and u.
%  Outputs perm, such that Sig(perm,perm)=L*L'.
%
% Reference:
%  Gibson G. J., Glasbey C. A., Elston D. A. (1994),
%  "Monte Carlo evaluation of multivariate normal integrals and
%  sensitivity to variate ordering",
%  In: Advances in Numerical Methods and Applications, pages 120--126

d=length(l);perm=1:d; % keep track of permutation
L=zeros(d,d);z=zeros(d,1);
for j=1:d
    pr=Inf(d,1); % compute marginal prob.
    I=j:d; % search remaining dimensions
    D=diag(Sig);
    s=D(I)-sum(L(I,1:j-1).^2,2);
    s(s<0)=eps;s=sqrt(s);
    tl=(l(I)-L(I,1:j-1)*z(1:j-1))./s;
    tu=(u(I)-L(I,1:j-1)*z(1:j-1))./s;
    pr(I)=lnNpr(tl,tu);
    % find smallest marginal dimension
    [dummy,k]=min(pr);
    % flip dimensions k-->j
    jk=[j,k];kj=[k,j];
    Sig(jk,:)=Sig(kj,:);Sig(:,jk)=Sig(:,kj); % update rows and cols of Sig
    L(jk,:)=L(kj,:); % update only rows of L
    l(jk)=l(kj);u(jk)=u(kj); % update integration limits
    perm(jk)=perm(kj); % keep track of permutation
    % construct L sequentially via Cholesky computation
    s=Sig(j,j)-sum(L(j,1:j-1).^2);
    if s<-0.01
        error('Sigma is not positive semi-definite')
    end
    s(s<0)=eps;L(j,j)=sqrt(s);
    L(j+1:d,j)=(Sig(j+1:d,j)-L(j+1:d,1:j-1)*(L(j,1:j-1))')/L(j,j);
    % find mean value, z(j), of truncated normal:
    tl=(l(j)-L(j,1:j-1)*z(1:j-1))/L(j,j);
    tu=(u(j)-L(j,1:j-1)*z(1:j-1))/L(j,j);
    w=lnNpr(tl,tu); % aids in computing expected value of trunc. normal
    z(j)=(exp(-.5*tl.^2-w)-exp(-.5*tu.^2-w))/sqrt(2*pi);
end
end



