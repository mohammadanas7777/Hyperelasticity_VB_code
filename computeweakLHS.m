function[LHS_weak] = computeweakLHS(NN,NDIM,Q,NE,NEN,gradNa,dQdI1,dQdI3,dI1dF,dI3dF,qpweights,matrix_global)
LHS_weak = zeros(NN*NDIM,size(Q,2));
for i = 1:NE
    B_element  =  [gradNa(i,1),0,gradNa(i,3),0,gradNa(i,5),0;gradNa(i,2),0,gradNa(i,4),0,gradNa(i,6),0;0,gradNa(i,1),0,gradNa(i,3),0,gradNa(i,5);0,gradNa(i,2),0,gradNa(i,4),0,gradNa(i,6)];
    d_features_dF_element = dQdI1(i,:)'*dI1dF(i,:)+dQdI3(i,:)'*dI3dF(i,:);
    LHS_element = qpweights(i).*(B_element'*d_features_dF_element');
    for k = 1:NEN*NDIM
        l = matrix_global(i,k);
        LHS_weak(l,:) = LHS_weak(l,:)+LHS_element(k,:);
    end
end
LHS_weak;
end