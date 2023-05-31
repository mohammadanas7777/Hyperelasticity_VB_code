function out_vb = BVS_disspI_vb_StudBeta(X, y, initz0, tol, verbosity)
% This function implements VB for a Student's-t slab with beta-distribution p0
% The slab variance scales with the measurement variance
if(isempty(X) || isempty(y))
    error('X and or y is missing');
end
if(size(X,1) ~= size(y,1))
    error('Number of observations do not match');
end

X = normalize(X);           % Standardise the columns (mean 0 and sd 1)
X = [ones(size(X,1),1), X]; % Add a constant vector of ones
y = y - mean(y);            % Detrend the measurements

% Prior parameters of noise variance (Inverse Gamma dist)
A = 1e-4; B = 1e-4;	
% Prior parameters of slab variance (Inverse Gamma dist)
nu = 1; s2 = 1;
av = nu/2; bv = nu*s2/2;
% Prior parameters of p0
ap = 0.1; bp = 1;
A0z = eye(size(X,2));

% initz = [1;initz0(:)]; % Adding the intercept indicator variable (slightly less than 1 to prevent log(0) values)
% DS = run_VB_john(X, y, vs, A, B, tau0, p0, initz, tol, verbosity);

if(isempty(initz0))
    error('No initial value of z found');
else
    initz = [1;initz0(:)]; % Adding the intercept indicator variable 
    DS = run_VB_StudBeta(X, y, A, B, av, bv, ap, bp, A0z, initz, tol, verbosity);
end

out_vb   = DS;
modelIdx = find(DS.zmean > 0.5)';
modelIdx = setdiff(modelIdx,1);
out_vb.modelIdx = modelIdx-1;
out_vb.Zmed = DS.zmean(modelIdx)';
out_vb.Wsel = DS.wmean(modelIdx)';
out_vb.Wcov = DS.wCOV(modelIdx, modelIdx);
out_vb.sig2 = DS.sig2;

end

function [DS, LLcvg] = run_VB_StudBeta(Xc, yc, A, B, av, bv, ap, bp, A0z, initz, tol, verbosity)
% This function is the implementation of VB from John T. Ormerod paper (2014)
% This implementation does not use slab scaling by noise variance
% A,B   : constants of the IG prior over noise variance
% av,bv : prior parameters of IG prior over slab variance
% ap,bp : prior parameters of Beta prior over p0
% A0z   : Slab covariance multiplier
% initz : Initial value of z
% Xc    : Centered and standardized dictionary except the first column
% yc    : Centered observations 

% lambda    = logit(p0);

converged = 0;
iter      = 0;
max_iter  = 100;
LL        = zeros(max_iter,1); 

zm      = initz(:);
tausig  = 1000;
tauv    = 0.1;


X = Xc;
y = yc;
XtX     = X'*X;
XtX     = 0.5*(XtX + XtX');
Xty     = X'*y;
yty     = y'*y;
eyep    = eye(size(XtX));
[N,p]   = size(X); 
allidx  = 1:p;
invA0z  = A0z\eyep;
detA0z  = det(A0z);

% Store this sum outside of loop
halfp     = 0.5*p;
halfsumNp = 0.5*N + halfp;
termstore = 0.5*p - 0.5*N*log(2*pi) + A*log(B) - gammaln(A)...
            + av*log(bv) - gammaln(av) - betaln(ap,bp) - 0.5*log(detA0z);

zm(1)   = 1;    % Always include the intercept 

while (converged==0)
    iter = iter+1;
   
    Zm  = diag(zm);
    Omg = zm*zm' + Zm*(eyep-Zm);
    
    % Update the mean and covariance of the coefficients given mean of z
    term1    = XtX .* Omg;  
    term10   = term1 + tauv*invA0z;
    invSigma = tausig*(term10);
    Sigma    = invSigma \ eyep;
    mu       = tausig * Sigma * Zm * Xty;
    
    % Update tausig
    Abar     = A + halfsumNp;
    term2    = 2 * Xty' * Zm * mu;
    term3    = mu*mu' + Sigma;
    term4    = yty - term2 + trace(term10*term3);
    Bbar     = B + 0.5*term4; 
    if(Bbar<0)
        warning('s turned out be less than 0. Taking absolute value');
        Bbar = B + 0.5*abs(term4);
    end
    tausig   = Abar/Bbar;
    
    % Update tauv
    term5    = trace(term3*invA0z);
    avbar    = av + halfp;
    bvbar    = bv + 0.5 * tausig * term5;
    tauv     = avbar/bvbar;
    
    % Update parameters of q(p0)
    sz       = sum(zm);
    apbar    = ap + sz;
    bpbar    = bp + p - sz;
    
    % Update zm
    zstr   = zm;
    lambda = psi(apbar) - psi(bpbar); 
    
    order   = setdiff(randperm(p), 1, 'stable');
    for j = order
        muj     = mu(j);
        sigmaj  = Sigma(j,j);
        
        remidx  = setdiff(allidx,j);
        mu_j    = mu(remidx);
        Sigma_jj= Sigma(remidx,j);
        
        etaj    = lambda - 0.5*tausig*(muj^2 + sigmaj)*XtX(j,j) + tausig*X(:,j)'*...
                  (y*muj - X(:,remidx)*diag(zstr(remidx))*(mu_j*muj + Sigma_jj));
        
        zstr(j) = expit(etaj);
    end
    
    zm = zstr;
    
    % Calculate ELBO
    zmoneminus = 1 - zm;
    LL(iter) = termstore - Abar*log(Bbar) + gammaln(Abar)...
        - avbar*log(bvbar) + gammaln(avbar) + betaln(apbar, bpbar)...
        + 0.5*log(det(Sigma)) - nansum( zm.*log(zm) ) ...
        - nansum( zmoneminus.*log(zmoneminus) )...
        + 0.5*(tausig*tauv)*term5;
    
    if(verbosity)
        fprintf('Iteration = %d    log(Likelihood) = %f\n', [iter, LL(iter,1)]);
    end
    if(iter>1)
        cvg = (LL(iter,1) - LL(iter-1,1));
        if(cvg<0 && verbosity)
            disp('OOPS!  log(like) decreasing!!')
        elseif(cvg<tol || iter>max_iter)
            converged = 1;
            LL = LL(1:iter);
        end
    end    
    
end

DS.zmean = zm;
DS.wmean = mu;
DS.wCOV  = Sigma;
DS.sig2  = 1/tausig;
LLcvg    = LL(end);
end

function logitC = logit(C)
    logitC = log(C) - log(1-C);
end

function expitC = expit(C)
    expitC = 1./(1 + exp(-C));
end

% function out = logdet(A)
%     out  = 2*sum(log(diag(rchol(A))));
% end

