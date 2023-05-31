% A function for computing the Cholesky factorisation,
%                    A = X.'X,
% where X is a square, upper triangular matrix and A a square,
% symmetric (that is, A = A.') and positive semi-definite matrix.
% 
% The usage is: X = rchol(A);
%
% Unlike many implementations of the Cholesky factorisation, this
% one copes with semi-definite A matrices (ones that have 
% some zero eigenvalues).
%
%   written by Brett Ninness, School of EE & CS
%              Adrian Wills   University of Newcastle
%        		              Australia.

% Copyright (C) Brett Ninness

function [A] = rchol(A)

A = triu(A);
n = size(A,1);

tol=n*eps;

if A(1,1) <= tol
 A(1,1:n) = 0;
else
 A(1,1:n) = A(1,1:n)/sqrt(A(1,1));
end

for j=2:n
 A(j,j:n) = A(j,j:n) - A(1:j-1,j)'*A(1:j-1,j:n);
 if A(j,j) <= tol
  A(j,j:n) = 0;
 else
  A(j,j:n) = A(j,j:n)/sqrt(A(j,j));
 end
end