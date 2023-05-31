function[LHS_ss,RHS_ss] = extendLHSRHS_SS(NN,bcs_nodes,LHS_weak,forces,balance)

% LHS_bulk = LHS (non-dirichlet nodes , :)
RHS_weak = LHS_weak;
for i = 1:NN
    if bcs_nodes(i,2) == 1
        LHS_weak(2*i-1,:)  =  LHS_weak(2*i-1,:).*balance;
        RHS_weak(2*i-1,:)  =  RHS_weak(2*i-1,:).*(balance*forces(1));
    end
    if bcs_nodes(i,2) == 2
        LHS_weak(2*i-1,:) = LHS_weak(2*i-1,:).*balance;
        RHS_weak(2*i-1,:) = RHS_weak(2*i-1,:).*(balance*forces(2));
    end
    if bcs_nodes(i,3) == 3
        LHS_weak(2*i,:) = LHS_weak(2*i,:).*balance;
        RHS_weak(2*i,:) = RHS_weak(2*i,:).*(balance*forces(3));
    end
    if bcs_nodes(i,3) == 4
        LHS_weak(2*i,:) = LHS_weak(2*i,:).*balance;
        RHS_weak(2*i,:) = RHS_weak(2*i,:).*(balance*forces(4));
    end
end

LHS_ss = LHS_weak;
RHS_ss = RHS_weak * ones(size(LHS_ss,2),1);

end
   
