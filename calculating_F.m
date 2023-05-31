function[F] = calculating_F(NE,NDIM,NEN,gradNa,u)
F = zeros(NE,NDIM*NDIM);
for i = 1:NE
    F_j = zeros(2);
    for j = 1:NEN
        if j ~= 1
            j = 2*j-1;
        end
        gradNa_j = [gradNa(i,j),gradNa(i,j+1)];
        u_j = [u(i,j),u(i,j+1)];
        F_j = F_j+u_j'*gradNa_j;
    end
    F_j = F_j+eye(size(F_j));
    F(i,:) = reshape(F_j,[1,4]);

end
v = F(:, 2);
F(:, 2) = F(:, 3);
F(:, 3) = v;
F;
end

