function[gradNa]=formgradientofShapefunction(NE,connectivity,x_nodes)
for i=1:NE
    l=connectivity(i,1);
    m=connectivity(i,2);
    n=connectivity(i,3);
    Ae=0.5*((x_nodes(m,2)*x_nodes(n,3)-x_nodes(n,2)*x_nodes(m,3))+(x_nodes(m,3)-x_nodes(n,3))*x_nodes(l,2)+(x_nodes(n,2)-x_nodes(m,2))*x_nodes(l,3));
    gradNa(i,:)=(1/(2*Ae)).*[(x_nodes(m,3)-x_nodes(n,3)),(x_nodes(n,2)-x_nodes(m,2)),(x_nodes(n,3)-x_nodes(l,3)),(x_nodes(l,2)-x_nodes(n,2)),(x_nodes(l,3)-x_nodes(m,3)),(x_nodes(m,2)-x_nodes(l,2))];
end
gradNa;
end