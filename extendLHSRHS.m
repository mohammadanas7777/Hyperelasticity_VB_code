function[LHS_step,RHS_step] = extendLHSRHS(NN,bcs_nodes,LHS_weak,forces,balance)

% LHS_bulk = LHS (non-dirichlet nodes , :)

j = 1;
for i = 1:NN
    if bcs_nodes(i,2) == 0
        LHS_bulk(j,:)  =  LHS_weak(2*i-1,:);
        j = j+1;
    end
    if bcs_nodes(i,3) == 0
        LHS_bulk(j,:)  =  LHS_weak(2*i,:);
        j = j+1;
    end
end
LHS_bulk  =  2.*LHS_bulk'*LHS_bulk;
RHS_reactions = zeros(size(LHS_bulk,1));

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

LHSa = LHSa'*ones(size(LHSa,1),1);
LHSb = LHSb'*ones(size(LHSb,1),1);
LHSc = LHSc'*ones(size(LHSc,1),1);
LHSd = LHSd'*ones(size(LHSd,1),1);
LHS_reactions = 2.*( LHSa*LHSa'+ LHSb*LHSb'+ LHSc*LHSc'+ LHSd*LHSd');
RHS_reactions = 2.*(LHSa*forces(1) + LHSb*forces(2) + LHSc*forces(3) + LHSd*forces(4));

LHS_step = LHS_bulk + balance*LHS_reactions;
RHS_step = balance*RHS_reactions;

end
   
