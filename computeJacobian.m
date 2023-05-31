function[J]=computeJacobian(NE,NDIM,F)
for i=1:NE
    l=1;
    for j=1:NDIM
        for k=1:NDIM
            Fnew(j,k)=F(i,l);
            l=l+1;
        end
    end
    J(i,1)=det(Fnew);
end
J;
end