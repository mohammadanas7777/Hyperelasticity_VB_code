function[qpweights] = gaussweights(x,NE)
for i = 1:NE
    x_element = reshape(x(i,:),3,2);
    [Wx] = triquad(1,x_element);
    qpweights(i,1) = Wx;
end