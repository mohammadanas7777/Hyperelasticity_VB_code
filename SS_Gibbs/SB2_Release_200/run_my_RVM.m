function [w_infer,SIGMA, model_RVM] = run_my_RVM(y, BASIS, n_iter)

    % Normalize the deisgn matrix
    X = normalize(BASIS);
    X = [X, ones(size(X,1),1)];
    y = y - mean(y);
    
    p = size(X,2)-1;

    noise_std = 1;
    likelihood_ = 'Gaussian';
    OPTIONS		= SB2_UserOptions('iterations',n_iter, 'diagnosticLevel', 0, 'monitor', 0);
    SETTINGS	= SB2_ParameterSettings('NoiseStd',noise_std);          
    
    [PARAMETER, HYPERPARAMETER, DIAGNOSTIC] = SparseBayes(likelihood_, X, y, OPTIONS, SETTINGS);
      
    Alpha = HYPERPARAMETER.Alpha;
    beta = HYPERPARAMETER.beta;
    USED = PARAMETER.Relevant;   
    
    PHI = X(:,USED);
    
    U	= chol(PHI'*PHI*beta + diag(Alpha));
    Ui	= inv(U);
    SIGMA	= Ui * Ui';
           
    oneidx = find(USED == (p+1));
    if(~isempty(oneidx))
        USED(oneidx) = [];
        relevant_idx = setdiff(1:length(USED), p+1);
          
        w_infer	= zeros(size(BASIS,2),1);
        w_infer(USED) = PARAMETER.Value(relevant_idx);
        SIGMA         = SIGMA(relevant_idx, relevant_idx);
        
        PARAMETER.Relevant = PARAMETER.Relevant(relevant_idx);
        PARAMETER.Value    = PARAMETER.Value(relevant_idx);
        
        DIAGNOSTIC.Gamma = DIAGNOSTIC.Gamma(relevant_idx);
    else
        w_infer	= zeros(size(BASIS,2),1);
        w_infer(USED) = PARAMETER.Value;
    end
    
    model_RVM.parameters = PARAMETER; 
    model_RVM.hyper      = HYPERPARAMETER;
    model_RVM.diagnostic = DIAGNOSTIC;
end
