function out_vb = BVS_disspI_vb_Rajdip2(X, y, initz0, tol, verbosity)
% This function implements VB for a Student's-t slab
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
N = size(X,1);

% Prior parameters of noise variance (Inverse Gamma dist)
A = 1e-4; B = 1e-4;	
% Prior parameters of slab variance (Inverse Gamma dist)
nu = 1; s2 = 0.5;
av = nu/2; bv = nu*s2/2;

% initz = [1;initz0(:)]; % Adding the intercept indicator variable (slightly less than 1 to prevent log(0) values)
% DS = run_VB_john(X, y, vs, A, B, tau0, p0, initz, tol, verbosity);

if(isempty(initz0))
    DS = output_VB(X, y, vs, A, B, tau0, [], tol, verbosity);
else
    p0set = expit(-5:0.05:0);
    initz = [1;initz0(:)]; % Adding the intercept indicator variable (slightly less than 1 to prevent log(0) values)
    J  = length(p0set);
    LL = nan(J,1);
    for j = 1:length(p0set)
        [~,LL(j)] = run_VB2(X, y, A, B, av, bv, p0set(j), initz, tol, false);
    end
    [~,k] = max(LL);
    DS = run_VB2(X, y, A, B, av, bv, p0set(k), initz, tol, verbosity);
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

function [DS, LLcvg] = run_VB2(Xc, yc, A, B, av, bv, p0, initz, tol, verbosity)
% This function is the implementation of VB from John T. Ormerod paper (2014)
% This implementation does not use slab scaling by noise variance
% A,B   : constants of the IG prior over noise variance
% av,bv : prior parameters of IG prior over slab variance
% p0    : inclusion probability
% initz : Initial value of z
% Xc    : Centered and standardized dictionary except the first column
% yc    : Centered observations 

lambda    = logit(p0);

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

% Store this sum outside of loop
halfp     = 0.5*p;
halfsumNp = 0.5*N + halfp;
termstore = 0.5*p - 0.5*N*log(2*pi) + A*log(B) - gammaln(A)...
            + av*log(bv) - gammaln(av);

zm(1)   = 1;    % Always include the intercept 

while (converged==0)
    iter = iter+1;
   
    Zm  = diag(zm);
    Omg = zm*zm' + Zm*(eyep-Zm);
    
    % Update the mean and covariance of the coefficients given mean of z
    term1    = XtX .* Omg;  
    invSigma = tausig*term1 + tauv*eyep;
    Sigma    = invSigma \ eyep;
    mu       = tausig * Sigma * Zm * Xty;
    
    % Update tausig
    Abar     = A + halfsumNp;
    term2    = 2 * Xty' * Zm * mu;
    term3    = mu*mu' + Sigma;
    term4    = yty - term2 + trace(term1*term3);
    Bbar     = B + 0.5*term4; 
    if(Bbar<0)
        warning('s turned out be less than 0. Taking absolute value');
        Bbar = B + 0.5*abs(term4);
    end
    tausig   = Abar/Bbar;
    
    % Update tauv
    avbar    = av + halfp;
    bvbar    = bv + 0.5 * tausig * trace(mu*mu' + Sigma);
    tauv     = avbar/bvbar;
      
    % Update zm
    zstr   = zm;
    
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
    
    % Calculate marginal log-likelihood
    zmoneminus = 1 - zm;
    LL(iter) = termstore - Abar*log(Bbar) + gammaln(Abar)...
        - avbar*log(bvbar) + gammaln(avbar) + 0.5*log(det(Sigma)) ...
        + nansum( zm.*(log(p0) - log(zm)) ) ...
        + nansum( zmoneminus.*(log(1-p0) - log(zmoneminus)) );
    
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

