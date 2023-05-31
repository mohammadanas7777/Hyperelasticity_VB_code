function[I1,I2,I3]=computeStrainInvariants(NE,NDIM,C)
for i=1:NE
    l=1;
    for j=1:NDIM
        for k=1:NDIM
            Cnew(j,k)=C(i,l);
            l=l+1;
        end
    end
    I1(i,1)=trace(Cnew)+1;
%     I2(i,1)=Cnew(1,1)+Cnew(2,2)-Cnew(1,2)*Cnew(2,1)+Cnew(1,1)*Cnew(2,2);
    I2(i,1) = 0.5*(I1(i,1)^2 - (trace(Cnew^2)+1));
    I3(i,1)=det(Cnew);
end
I1;
I2;
I3;
end
