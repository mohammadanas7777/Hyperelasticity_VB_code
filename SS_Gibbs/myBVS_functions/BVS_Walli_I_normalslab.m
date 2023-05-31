function out = BVS_Walli_I_normalslab(X, y, nSamples, nburn, nchains, verbosity)
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
    vs_i        = vs;
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
alpha0      = 1000;
beta0       = 1000;

[n,d]       = size(X);

% Perform some initializations
samplesSig2 = zeros(nSamples, 1);   % Sigma^2 (noise variance)
samplesZ    = zeros(nSamples, d);
samplesW    = zeros(nSamples, d);

samplesSig2(1)  = 1/noiseBeta;
samplesZ(1,:)   = Z0;
samplesW(1,:)   = W0;

invVs           = 1/vs;
A0              = eye(d);
% Start sampling from the conditional distributions
for i = 2:nSamples
    
    prevZ           = samplesZ(i-1,:);
    sz              = sum(prevZ);
    
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
        pratio = p0/(p0 + exp(log_ml_yz0_vs - log_ml_yz1_vs)*(1-p0) );
    
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
DS.samplesSig2 = samplesSig2;

DS.prior.sig2  = [alpha0, beta0];

end

%% MARGINAL LOG-LIKELIHOOD for sampling z (following Walli-Malsiner 2010)
function logML = marlike(X, y, Z, invVs, A0, alpha0, beta0) 

N       = size(X,1);    % number of observations

sz      = sum(Z);       % number of active z-variables
eyesz   = eye(sz);
indx    = find(Z==1);   % Finding indices of Z which are non-zero
XXz     = X(:,indx);
A0z     = A0(indx, indx);
invAz   = XXz'*XXz + invVs*(A0z\eyesz);
invAz   = 0.5*(invAz + invAz');
Az      = invAz \ eyesz;

XXty    = XXz'*y;
SN      = 0.5*(y'*y - XXty'*Az*XXty);

logML   = -0.5*N*log(2*pi) + 0.5*logdet(Az) - 0.5*logdet(A0z) + alpha0*log(beta0)...
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
%     Sigmaz = Sigmaz'*Sigmaz;
    meanW           = Sigmaz*Xz'*y;         % a_delta
    muW(idx,1)      = meanW;
%     newW(:,idx)     = meanW' + randn(1,r)*cholcov(noiseBeta\Sigmaz);    
%     R = rmvnrnd(meanW,(0.5).*((noiseBeta\Sigmaz)+(noiseBeta\Sigmaz)'),r,(-1).*eye(r),ones(r,1));
    D = (diag(noiseBeta\Sigmaz));
    R=zeros(1,r);
    for i = 1:r
        R(1,i) = meanW(i)+D(i)*trandn((0-meanW(i))/D(i),inf);
    end
    newW(:,idx) = R(1,:);
%     newW(:,idx)     = mvnrnd(meanW',noiseBeta\Sigmaz);
%     newW(:,idx)     = rmvnrnd(repmat(meanW,1,1),cholcov(noiseBeta\Sigmaz),zeros(1,r),ones(1,r));

%     pd = makedist('normal');
%     tr = truncate(pd,0,inf);
%     rr = random(tr,1,r);
%     newW(:,idx)     = meanW' + rr*cholcov(noiseBeta\Sigmaz);

%     sig = eig(noiseBeta\Sigmaz);
%     for g = 1:r
%         pd = makedist('normal','mu',meanW(g),'sigma',sig(g));
%         tr = truncate(pd,0,inf);
%         newW(:,idx(g)) = random(tr,1,1);
%     end


   
    
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

function [X] = rmvnrnd(mu,sigma,N,A,b)
%RMVNRND Draw from the  truncated multivariate normal distribution.
%   X = rmvnrnd(MU,SIG,N,A,B) returns in N-by-P matrix X a
%   random sample drawn from the P-dimensional multivariate normal
%   distribution with mean MU and covariance SIG truncated to a
%   region bounded by the hyperplanes defined by the inequalities Ax<=B. If
%   A and B are ommitted then the sample is drawn from the unconstrained
%   MVN distribution.
%
%   [X,RHO,NAR,NGIBBS]  = rmvnrnd(MU,SIG,N,A,B) returns the
%   acceptance rate RHO of the accept-reject portion of the algorithm
%   (see below), the number NAR of returned samples generated by
%   the accept-reject algorithm, and the number NGIBBS returned by
%   the Gibbs sampler portion of the algorithm.
%
%   rmvnrnd(MU,SIG,N,A,B,RHOTHR) sets the minimum acceptable
%   acceptance rate for the accept-reject portion of the algorithm
%   to RHOTHR. The default is the empirically identified value
%   2.9e-4.
%
%   ALGORITHM
%   The function begins by drawing samples from the untruncated MVN
%   distribution and rejecting those which fail to satisfy the
%   constraints. If, after a number of iterations with escalating
%   sample sizes, the acceptance rate is less than RHOTHR it
%   switches to a Gibbs sampler. 
%
% ACKNOWLEDGEMENT
%   This makes use of TruncatedGaussian by Bruno Luong (File ID:
%   #23832) to generate draws from the one dimensional truncated normal.
%
%   REFERENCES
%   Robert, C.P, "Simulation of truncated normal variables",
%   Statistics and Computing, pp. 121-125 (1995).




% Copyright 2011 Tim J. Benham, School of Mathematics and Physics,
%                University of Queensland.

%
% Constant parameters
%
defaultRhoThr = 2.9e-4;                   % min. acceptance rate to
                                          % accept accept-reject sampling.
%
% Process input arguments.
%
if nargin<6 || rhoThr<0,rhoThr = defaultRhoThr;
end

% If no constraints default to mvnrnd
if nargin<5
    rho = 1; nar = N; ngibbs = 0;
    X = mvnrnd(mu, sigma, N);
    return
end
    
A = A'; b = b';
mu = mu';

p = length(mu);                         % dimension
m = size(A,2);                          % number of constraints
if m==0
    A = zeros(p,1);
end

% initialize return arguments
X = zeros(N,p);                         % returned random variates
nar = 0;                                % no. of X generated by a-r
ngibbs = 0;
rho = 1; 

if rhoThr<1
    % Approach 1: accept-reject method
    n = 0; % no. accepted
    maxSample = 1e6;
    trials = 0; passes = 0;
    s = N;
    while n<N && ( rho>rhoThr || s<maxSample)
        R  = mvnrnd(mu,sigma,s);
        R = R(sum(R*A<=repmat(b,size(R,1),1),2)==size(A,2),:);
        if size(R,1)>0
            X((n+1):min(N,(n+size(R,1))),:) = R(1:min(N-n,size(R,1)),:);
            nar = nar + min(N,(n+size(R,1))) - n;
        end
        n = n + size(R,1); trials = trials + s;
        rho = n/trials;
        if rho>0
            s = min([maxSample,ceil((N-n)/rho),10*s]);
        else
            s = min([maxSample,10*s]);
        end
        passes = passes + 1;
    end
end

%
% Approach 2: Gibbs sampler of Christian Robert
%
if nar < N
    % choose starting point
    if nar > 0
        x = X(nar,:);
        discard = 0;
    elseif all(A'*mu' <= b')
        x = mu';
        discard = 1;
    else
        % need to find a feasible point
        xf = chebycenter(A',b',1);
        if ~all(A'*xf <= b') 
            error('Failed to find a feasible point')
        end
        % would like a feasible point near mu
        Atheta = -A'*(xf-mu');
        btheta = b' - A'*xf;
        Atheta = [Atheta; 1; -1];
        btheta = [btheta; 1; 0 ];
        ftheta = -1;
        % x = linprog(f,A,b,Aeq,beq,lb,ub,x0,options)
        options = optimset;
        options = optimset(options,'Display', 'off');
        theta = linprog(ftheta,Atheta,btheta,[],[],[],[],[],options);
        x = mu' + (1-theta)*(xf-mu');
        
        discard = 1;
    end
    % set up inverse Sigma
    SigmaInv = inv(sigma);
    n = nar;
    while n<N
        % choose p new components
        for i = 1:p
            % Sigmai_i is the (p − 1) vector derived from the i-th column of Σ
            % by removing the i-th row term.
            Sigmai_i = sigma([1:(i-1) (i+1):p],i);
            % Sigma_i_iInv is the inverse of the (p−1)×(p−1) matrix
            % derived from Σ = (σij ) by eliminating its i-th row and
            % its i-th column 
            Sigma_i_iInv = SigmaInv([1:(i-1) (i+1):p],[1:(i-1) (i+1):p]) - ...
                SigmaInv([1:(i-1) (i+1):p],i)*SigmaInv([1:(i-1) (i+1):p],i)' ...
                / SigmaInv(i,i);
            % x_i is the (p-1) vector of components not being updated
            % at this iteration. /// mu_i
            x_i = x([1:(i-1) (i+1):p])';
            mu_i = mu([1:(i-1) (i+1):p]);
            % mui is E(xi|x_i)
            %        mui = mu(i) + Sigmai_i' * Sigma_i_iInv * (x_i - mu_i);
            mui = mu(i) + Sigmai_i' * Sigma_i_iInv * (x_i' - mu_i');
            s2i = sigma(i,i) - Sigmai_i'*Sigma_i_iInv*Sigmai_i;
            % Find points where the line with the (p-1) components x_i
            % fixed intersects the bounding polytope.
            % A_i is the (p-1) x m matrix derived from A by removing
            % the i-th row.
            A_i = A([1:(i-1) (i+1):p],:);
            % Ai is the i-th row of A
            Ai = A(i,:);
            c = (b-x_i*A_i)./Ai;
            lb = max(c(Ai<0));
            if isempty(lb), lb=-Inf; end
            ub = min(c(Ai>0));
            if isempty(ub), ub=Inf; end
            
            %% test feasibility of using TruncatedGaussian
            % lbsig = (lb-mui)/sqrt(s2i);
            % ubsig = (ub-mui)/sqrt(s2i);
            % if lbsig > 10 || ubsig < -10
            %     fprintf('YOWZA! %.2f %.2f %.2f %.2f\n', lbsig, ubsig, ...
            %             mui, sqrt(s2i));
            % end
            %%
            
            % now draw from the 1-d normal truncated to [lb, ub]
            x(i) = mui+TruncatedGaussian(-sqrt(s2i),[lb ub]-mui);
        end
        if discard <= 0
            n = n + 1;
            X(n,:) = x; 
            ngibbs = ngibbs+1;
        end
        discard = discard-1;
    end
end
end
function  rv=mvrandn2(l,u,Sig,n)
%% truncated multivariate normal generator
% simulates 'n' random vectors exactly/perfectly distributed
% from the d-dimensional N(0,Sig) distribution (zero-mean normal
% with covariance 'Sig') conditional on l<X<u;
% infinite values for 'l' and 'u' are accepted;
% output:   'd' times 'n' array 'rv' storing random vectors;
%
% * Example:
%  d=60;n=10^3;Sig=0.9*ones(d,d)+0.1*eye(d);l=(1:d)/d*4;u=l+2;
%  X=mvrandn(l,u,Sig,n);boxplot(X','plotstyle','compact') % plot marginals
%
% * Notes: Algorithm may not work if 'Sig' is close to being rank deficient.

% Reference:
% Z. I. Botev (2015), "The Normal Law Under Linear Restrictions:
%  Simulation and Estimation via Minimax Tilting", submitted to JRSS(B)
l=l(:); u=u(:); % set to column vectors
d=length(l); % basic input check
if  (length(u)~=d)|(d~=sqrt(prod(size(Sig)))|any(l>u))
    error('l, u, and Sig have to match in dimension with u>l')
end
% Cholesky decomposition of matrix with permuation
[Lfull,l,u,perm]=cholperm(Sig,l,u); % outputs the permutation
D=diag(Lfull);
if any(D<eps)
    warning('Method may fail as covariance matrix is singular!')
end
L=Lfull./repmat(D,1,d);u=u./D; l=l./D; % rescale
L=L-eye(d); % remove diagonal
% find optimal tilting parameter non-linear equation solver
options=optimset('Diagnostics','off','Display','off',...
    'Algorithm','trust-region-dogleg','Jacobian','on');
[soln,fval,exitflag] = fsolve(@(x)gradpsi(x,L,l,u),zeros(2*(d-1),1),options);
if exitflag~=1
    warning('Method may fail as covariance matrix is close to singular!')
end
x=soln(1:(d-1));mu=soln(d:(2*d-2));
% compute psi star
psistar=psy(x,L,l,u,mu);
% start acceptance rejection sampling
rv=[]; accept=0; iter=0;
while accept<n % while # of accepted is less than n
    [logpr,Z]=mvnrnd(n,L,l,u,mu); % simulate n proposals
    idx=-log(rand(1,n))>(psistar-logpr); % acceptance tests
    rv=[rv,Z(:,idx)];  % accumulate accepted
    accept=size(rv,2); % keep track of # of accepted
    iter=iter+1;  % keep track of while loop iterations
    if iter==10^3 % if iterations are getting large, give warning
        warning('Acceptance prob. smaller than 0.001')
    elseif iter>10^4 % if iterations too large, seek approximation only
        accept=n;rv=[rv,Z]; % add the approximate samples
        warning('Sample is only approximately distributed.')
    end
end
% finish sampling; postprocessing
[dum,order]=sort(perm,'ascend');
rv=rv(:,1:n); % cut-down the array to desired n samples
rv=Lfull*rv; % reverse scaling of L
rv=rv(order,:); % reverse the Cholesky permutation
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [p,Z]=mvnrnd(n,L,l,u,mu)
% generates the proposals from the exponentially tilted
% sequential importance sampling pdf;
% output:    'p', log-likelihood of sample
%             Z, random sample
d=length(l); % Initialization
mu(d)=0;
Z=zeros(d,n); % create array for variables
p=0;
for k=1:d
    % compute matrix multiplication L*Z
    col=L(k,1:k)*Z(1:k,:);
    % compute limits of truncation
    tl=l(k)-mu(k)-col;
    tu=u(k)-mu(k)-col;
    %simulate N(mu,1) conditional on [tl,tu]
    Z(k,:)=mu(k)+trandn(tl',tu');
    % update likelihood ratio
    p = p+lnNpr(tl,tu)+.5*mu(k)^2-mu(k)*Z(k,:);
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [grad,J]=gradpsi(y,L,l,u)
% implements gradient of psi(x) to find optimal exponential twisting;
% assumes scaled 'L' with zero diagonal;
d=length(u);c=zeros(d,1);x=c;mu=c;
x(1:(d-1))=y(1:(d-1));mu(1:(d-1))=y(d:end);
% compute now ~l and ~u
c(2:d)=L(2:d,:)*x;lt=l-mu-c;ut=u-mu-c;
% compute gradients avoiding catastrophic cancellation
w=lnNpr(lt,ut);
pl=exp(-0.5*lt.^2-w)/sqrt(2*pi);
pu=exp(-0.5*ut.^2-w)/sqrt(2*pi);
P=pl-pu;
% output the gradient
dfdx=-mu(1:(d-1))+(P'*L(:,1:(d-1)))';
dfdm= mu-x+P;
grad=[dfdx;dfdm(1:(d-1))];
if nargout>1 % here compute Jacobian matrix
    lt(isinf(lt))=0; ut(isinf(ut))=0;
    dP=-P.^2+lt.*pl-ut.*pu; % dPdm
    DL=repmat(dP,1,d).*L;
    mx=-eye(d)+DL;
    xx=L'*DL;
    mx=mx(1:d-1,1:d-1);
    xx=xx(1:d-1,1:d-1);
    J=[xx,mx';
        mx,diag(1+dP(1:d-1))];
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function p=psy(x,L,l,u,mu)
% implements psi(x,mu); assumes scaled 'L' without diagonal;
d=length(u);x(d)=0;mu(d)=0;x=x(:);mu=mu(:);
% compute now ~l and ~u
c=L*x;l=l-mu-c;u=u-mu-c;
p=sum(lnNpr(l,u)+.5*mu.^2-x.*mu);
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


