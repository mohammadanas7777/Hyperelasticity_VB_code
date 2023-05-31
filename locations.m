function[x]=locations(x_nodes,connectivity)
%u=zeros(length(connectivity),6);
for i=1:length(connectivity)
    x(i,1)=x_nodes(connectivity(i,1),2);
    x(i,2)=x_nodes(connectivity(i,2),2);
    x(i,3)=x_nodes(connectivity(i,3),2);
    x(i,4)=x_nodes(connectivity(i,1),3);
    x(i,5)=x_nodes(connectivity(i,2),3);
    x(i,6)=x_nodes(connectivity(i,3),3);
end
x;
end