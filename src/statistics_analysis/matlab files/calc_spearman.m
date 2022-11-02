% CALC_SPEARMAN     compute Spearman's correlation coeffiecient.
%
% call              [ cc, pval ] = calc_spearman( x, y, nreps )
%
% gets              x       data matrix (2 columns)
%                   y       optional; then computes the CC between columns of x and y
%                   nreps   optional; then conducts a permutation test, nreps times
%
% returns           cc      correlation coefficient
%                   pval    p-value
%
% calls             calc_pearson, mixmat
%
% reference         Sokal and Rohlf 2001, p. 593-600
%
% usage             when x,y do not conform to a bivariate normal distribution (p.593)

% Spearman's cc is better than Kendall's rank correlation when there is low
% certainty about reliability of close ranks (p. 600; i.e. there are
% errors in measurements). 
% for n<=10, significance testing requires special tables, and for n > 10,
% can be tested in a similar manner as Pearson's cc (p.600). this routine
% implements a general and assumption-free approach for significance
% testing (a permutation test)

% 12-jul-04 ES

% last update
% August 2022

function [ cc, pval ] = calc_spearman( x, y, nreps )

% intialize output
cc                              = [];
pval                            = [];

% handle arguments
if nargin < 1 || isempty( x )
    return
end
if nargin < 2
    y                           = [];
end
if nargin < 3 || isempty( nreps )
    nreps                       = 0;
end

% handle NaNs
nansX                           = isnan( x );
if isempty( y )
    nans                        = sum( nansX, 2 ) > 0;
    x                           = x( ~nans, : );
elseif isequal( size( x ), size( y ) ) && size( x, 2 ) == 1
    nans                        = sum( isnan( y ) | nansX, 2 ) > 0;
    x                           = x( ~nans, : );
    y                           = y( ~nans, : );
else
    % ignore NaNs for now, handled in calc_pearson
end

% rank-order and compute
x                               = local_rankcols( x );
y                               = local_rankcols( y );
cc                              = calc_pearson( x, y );

% prepare for significance testing
if nreps == 0 || isempty( x )
    if isempty( x ) && all( nans( : ) )
        cc                      = NaN;
    end
    pval                        = NaN;
    return
end
if isempty( y )                                 % simple case #1
    x1                          = x( :, 1 );
    x2                          = x( :, 2 );
elseif size( x, 2 ) == 1 && size( y, 2 ) == 1   % simple case #2
    x1                          = x;
    x2                          = y;
elseif size( x, 2 ) == size( y, 2 )             % recursion
    
    npairs                      = size( x, 2 );
    cc                          = NaN * ones( 1, npairs );
    pval                        = cc;
    for i               = 1 : npairs
        if all( isnan( x( :, i ) ) ) || all( isnan( y( :, i ) ) )
            cc( i )             = NaN;
            pval( i )           = NaN;
        else
            [ cc( i ), pval( i ) ] = calc_spearman( x( :, i ), y( :, i ), nreps );
        end
    end
    return
else                                            % aberrant matrices
    pval                        = NaN * ones( size( cc ) );
    return
end

% estimate significance
v                               = ones( 1, nreps );
ccmix                           = calc_spearman( x1 * v, mixmat( x2 * v, 1 ) );
pval                            = ( sum( abs( cc ) <= abs( ccmix ) ) + 1 ) / ( nreps + 1 );

return % calc_spearman

%------------------------------------------------------------------------
% y = local_rankcols( x )
% rank matrix column-wise
%------------------------------------------------------------------------
function y = local_rankcols( x )

[ m, n ]                        = size( x );
if m == 1
    x                           = x';
    m                           = n;
    n                           = 1;
end
nans                            = isnan( x );
ridx                            = m : -1 : 1;
cidx                            = 1 : n;
[ ~, idx ]                      = sort( [ x x( ridx, : ) ] );
[ ~, ranks ]                    = sort( idx );
y                               = ( ranks( :, cidx ) + ranks( ridx, cidx + n ) ) / 2;
y( nans )                       = NaN;

return % local_rankcols

% EOF
