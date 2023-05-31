function[u]=displacements(u_nodes,connectivity)
%u=zeros(length(connectivity),6);
for i=1:length(connectivity)
    u(i,1)=u_nodes(connectivity(i,1),2);
    u(i,2)=u_nodes(connectivity(i,1),3);
    u(i,3)=u_nodes(connectivity(i,2),2);
    u(i,4)=u_nodes(connectivity(i,2),3);
    u(i,5)=u_nodes(connectivity(i,3),2);
    u(i,6)=u_nodes(connectivity(i,3),3);
end
u;
end
