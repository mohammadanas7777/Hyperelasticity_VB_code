function[C]=computeCauchyGreenTensor(NE,NDIM,F)
for i=1:NE
    l=1;
    for j=1:NDIM
        for k=1:NDIM
            Fnew(j,k)=F(i,l);
            l=l+1;
        end
    end
    c=Fnew'*Fnew;
    C(i,:)=reshape(c.',[1,4]);
end
C;
end
