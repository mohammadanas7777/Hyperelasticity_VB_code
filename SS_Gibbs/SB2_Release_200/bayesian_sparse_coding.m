function [w_infer,SIGMA,logML, model] = bayesian_sparse_coding(x, BASIS, n_iter)



    noise_std = 1;
    likelihood_='Gaussian';
    OPTIONS		= SB2_UserOptions('iterations',n_iter,...
                  'diagnosticLevel', 0,...
                  'monitor', 0);
    SETTINGS	= SB2_ParameterSettings('NoiseStd',noise_std);          
    
    %                 basisWidth = 0.2;  
    %                 BASIS	= exp(-distSquared(Zp,Zp)/(basisWidth^2));
    
    
    
    %%
    
    LIKELIHOOD	= SB2_Likelihoods('Gaussian');
    [PARAMETER, HYPERPARAMETER, DIAGNOSTIC] = SparseBayes(likelihood_, BASIS, x, OPTIONS, SETTINGS);
    
    
    
%     USED = PARAMETER.Relevant;
%     PHI = BASIS(:,w_infer~=0);
%     beta = model.hyper.beta;
%     alpha = model.hyper.Alpha;
%     BASIS_PHI = BASIS'*PHI;
%     BASIS_TARGETS = BASIS'*yp;
% 
%     [SIGMA,MU,S_IN,Q_IN,S_OUT,Q_OUT,FACTOR,LOGML,GAMMA,...
%         BETABASIS_PHI,BETA] = ...
%      SB2_FullStatistics(LIKELIHOOD,BASIS,PHI,yp,USED,alpha,beta,...
%           w_infer,BASIS_PHI,BASIS_TARGETS,OPTIONS);
%       
    
      
    Alpha = HYPERPARAMETER.Alpha;
    beta = HYPERPARAMETER.beta;
    USED = PARAMETER.Relevant;
    
    
    
    PHI = BASIS(:,USED);
    
    U		= chol(PHI'*PHI*beta + diag(Alpha));
    Ui	= inv(U);
    SIGMA	= Ui * Ui';
    sigma_star =  PHI*SIGMA*PHI';
    
    w_infer	= zeros(size(BASIS,2),1);
    w_infer(USED) = PARAMETER.Value;
    Mu = PARAMETER.Value;
    
    
    %
    % Compute the inferred prediction function
    % 

    
    model.parameters = PARAMETER;
    model.hyper = HYPERPARAMETER;
    model.diagnostic = DIAGNOSTIC;
    
    
    e = (x-PHI*Mu);
    ED = e'*e;
    n = length(x);
    dataLikely	= (n*log(beta) - beta*ED)/2;
    
    logdetHOver2	= sum(log(diag(U)));
    logML			= dataLikely - (Mu.^2)'*Alpha/2 + ...
    sum(log(Alpha))/2 - logdetHOver2;
    
end
