function[LHS_free,LHS_fix,RHS_free,RHS_fix] = extendLHSRHS_ssnew(NN,bcs_nodes,LHS_weak,forces)

% LHS_free = LHS (non-dirichlet nodes , :)

j = 1;
for i = 1:NN
    if bcs_nodes(i,2) == 0
        LHS_free(j,:)  =  LHS_weak(2*i-1,:);
        j = j+1;
    end
    if bcs_nodes(i,3) == 0
        LHS_free(j,:)  =  LHS_weak(2*i,:);
        j = j+1;
    end
end

a = 1;
b = 1;
c = 1;
d = 1;
for i = 1:NN
    if bcs_nodes(i,2) == 1
        LHSa(a,:)  =  LHS_weak(2*i-1,:);
        a = a+1;
    end
    if bcs_nodes(i,2) == 2
        LHSb(b,:)  =  LHS_weak(2*i-1,:);
        b = b+1;
    end
    if bcs_nodes(i,3) == 3
        LHSc(c,:)  =  LHS_weak(2*i,:);
        c = c+1;
    end
    if bcs_nodes(i,3) == 4
        LHSd(d,:) = LHS_weak(2*i,:);
        d = d+1;
    end
end

LHSa = sum(LHSa,1);
LHSb = sum(LHSb,1);
LHSc = sum(LHSc,1);
LHSd = sum(LHSd,1);
LHS_fix = [LHSa;LHSb;LHSc;LHSd];
RHS_free = zeros(size(LHS_free,1),1);
RHS_fix = forces;

% LHS_step_ss = [LHS_free ; balance.*LHS_fix];
% RHS_step_ss = [RHS_free ; balance.*RHS_fix];

end