function out_vb = BVS_disspI_vb_John(X, y, initz0, tol, verbosity)
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

A = 1e-4; B = 1e-4;
vs = 10;
tau0 = 1000;    % Expected value of inverse of measurement variance

if(isempty(initz0))
    DS = output_VB(X, y, vs, A, B, tau0, [], tol, verbosity);
else
    p0 = expit(-0.5*sqrt(N));
    initz = [1;initz0(:)]; % Adding the intercept indicator variable (slightly less than 1 to prevent log(0) values)
    DS = run_VB(X, y, vs, A, B, tau0, p0, initz, tol, verbosity);
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

function out = output_VB(X, y, vs, A, B, tau0, initz, tol, verbosity)

[N,p]   = size(X);

if(isempty(initz))
    initz   = zeros(p,1);
end
initz(1)= 1; % Include the intercept

% initz([2,3,end]) = 1; % Include the variables of a linear system
p0      = expit(-0.5*sqrt(N));

P = 50;
M = 5;

LLmax = -1e+100;

LL = nan(p,1);
for i = 1:max(p,P)
    for j = 2:p
        initz1      = initz;
        initz1(j)   = 1;
        [~, LL(j)]  = run_VB(X, y, vs, A, B, tau0, p0, initz1, tol, false);
    end
    [maxLL, kk] = max(LL);    
    if(LLmax < maxLL)
        LLmax = maxLL;
        initz(kk)   = 1;
    else
        break
    end   
end

p0set = expit(-(0:0.5:5));
J     = length(p0set);
vals  = [0,1];
LLj   = nan(2,1);
for i = 1:M
    disp(num2str(i));
    LL = nan(J,1);
    for j = 1:J
        [~, LL(j)]  = run_VB(X, y, vs, A, B, tau0, p0set(j), initz, tol, false);
    end
    [maxLL, kk] = max(LL);  
    if(LLmax < maxLL)
        LLmax = maxLL;
        p0    = p0set(kk);
    end
    
    for j = 2:p
        initztmp = initz;
        initztmp(j) = vals(1);
        [~,LLj(1)]  = run_VB(X, y, vs, A, B, tau0, p0, initztmp, tol, false);
        initztmp(j) = vals(2);
        [~,LLj(2)]  = run_VB(X, y, vs, A, B, tau0, p0, initztmp, tol, false);
        
        [maxLL, kk] = max(LLj);
        if(LLmax < maxLL)
            LLmax = maxLL;
            initz(j) = vals(kk);
        end
    end
end

out = run_VB(X, y, vs, A, B, tau0, p0, initz, tol, verbosity);
        
end

function [DS, LLcvg] = run_VB(Xc, yc, vs, A, B, tau0, p0, initz, tol, verbosity)
% This function is the implementation of VB from John T. Ormerod paper (2014)
% This implementation does not use slab scaling by noise variance
% vs    : treated as a constant
% A,B   : constants of the IG prior over noise variance
% tau0  : Expected value of (sigma^{-2})
% p0    : inclusion probablility
% initz : Initial value of z
% Xc    : Centered and standardized dictionary except the first column
% yc    : Centered observations 

lambda = logit(p0);

converged = 0;
iter      = 0;
max_iter  = 100;
LL        = zeros(max_iter,1); 

zm      = initz(:);
taum    = tau0;
invVs   = 1/vs;

X = Xc;
y = yc;
XtX     = X'*X;
XtX     = 0.5*(XtX + XtX');
Xty     = X'*y;
yty     = y'*y;
eyep    = eye(size(XtX));
[N,p]   = size(X);
allidx  = 1:p;

zm(1)   = 1;    % Always include the intercept 

while (converged==0)
    iter = iter+1;
   
    Zm  = diag(zm);
    Omg = zm*zm' + Zm*(eyep-Zm);
    
    % Update the mean and covariance of the coefficients given mean of z
    term1    = XtX .* Omg;  
    invSigma = taum*term1 + invVs*eyep;
    Sigma    = invSigma \ eyep;
    mu       = taum * Sigma * Zm * Xty;
    
    % Update tau
    term2    = 2 * Xty' * Zm * mu;
    term3    = mu*mu' + Sigma;
    term4    = yty - term2 + trace(term1*term3);
    s        = B + 0.5*term4; 
    if(s<0)
        warning('s turned out be less than 0. Taking absolute value');
        s = B + 0.5*abs(term4);
    end
    taum     = (A + 0.5*N)/s;
    
    % Update zm
    zstr   = zm;
    
    order   = setdiff(randperm(p), 1, 'stable');
    for j = order
        muj     = mu(j);
        sigmaj  = Sigma(j,j);
        
        remidx  = setdiff(allidx,j);
        mu_j    = mu(remidx);
        Sigma_jj= Sigma(remidx,j);
        
        etaj    = lambda - 0.5*taum*(muj^2 + sigmaj)*XtX(j,j) + ...
                  taum*X(:,j)'*(y*muj - X(:,remidx)*diag(zstr(remidx))*(mu_j*muj + Sigma_jj));
        
        zstr(j) = expit(etaj);
    end
    
    zm = zstr;
    
    % Calculate marginal log-likelihood
    LL(iter) = 0.5*p - 0.5*N*log(2*pi) + 0.5*p*log(invVs) + A*log(B) - gammaln(A)...
        + gammaln(A+0.5*N) - (A+0.5*N)*log(s) + 0.5*log(det(Sigma))...
        - 0.5*invVs*trace(mu*mu' + Sigma) + nansum( zm.*(log(p0) - log(zm)) )... 
        + nansum( (1-zm).*(log(1-p0) - log(1-zm)) );  
    
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
DS.sig2  = 1/taum;
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

