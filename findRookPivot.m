% =========================================================================
% Algorithm: Rook Pivorting
%
% AUTHOR ..... [Alexander Railton, Matvey Ilyich Skripchenko, Andrew James]
% UPDATED .... [March 27 2018]
%
% takes matrix A as an input and returns the location of the pivot
%
% INPUT
% A ... a matrix of dimensions n x n
%
% OUTPUT
% [i,j]...The locations of the pivot
% =========================================================================
function [i,j] = findRookPivot(A)
    found = false;
    j = 1;
    
    while ~found
    
        while ~found
%             big = max(abs(A(:,j)));
            [big,i] = max(abs(A(:,j)));
%             p = max(abs(A(i,:)));
            [p, k] = max(abs(A(i,:)));
            
            if (p <= big) 
                found = true;
           
            else 
                j = k;
                
            end
        end

    end


end

