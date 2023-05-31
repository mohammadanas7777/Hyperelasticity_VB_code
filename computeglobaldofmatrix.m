function[matrix_global]=computeglobaldofmatrix(NE,NEN,connectivity,NDIM)
for i=1:NE
    for j=1:NEN
        NODEN=connectivity(i,j);
        for k=1:NDIM
            l=(j-1)*NDIM+k;
            matrix_global(i,l)=(NODEN-1)*NDIM+k;
        end
    end
end
end
