% CALC_PEARSON      compute Pearson's correlation coeffiecient.
%
% call              cc = calc_pearson( x )
%                   cc = calc_pearson( x, y )
%
% gets              x       data matrix (2 columns)
%                   y       optional; then, computes the CC between columns of X and Y
%
% returns           cc      correlation coefficient(s)
%
% calls             nothing.

% 22-mar-04 ES

% last update
% August 2019

function cc = calc_pearson( x, y )

if isempty( x ) && isempty( y )
    cc                          = [];
    return
end
[ m, n ]                        = size( x );
if m < 2
    cc                          = NaN * ones( 1, n );
    return
end
d                               = ones( m, 1 );

if nargin == 1 || isempty( y )                                              % correlation matrix (all possible x's column-pairs)
    
    v                           = ~isnan( x );
    vm                          = sum( v );
    y                           = x - d * ( nansum( x .* v ) ./ vm );
    s                           = sqrt( nansum( y .* y .* v ) );
    z                           = y ./ ( d * s );
    z( y == 0 )                 = 0;
    z( isnan( z ) )             = 0;
    cc                          = z' * z;
    if n == 2                                                               % correlation between columns (scalar)
        cc                      = cc( 1, 2 );
    end
    
elseif ~isequal( [ m n ], size( y ) )
    
    error( 'input size mismatch' )
    
else                                                                        % correlation vector (column pairs of x,y)
    
    v                       = ~isnan( x ) & ~isnan( y ); 
    vm                      = sum( v );
    x                       = x - d * ( nansum( x .* v ) ./ vm );
    y                       = y - d * ( nansum( y .* v ) ./ vm );
    num                     = nansum( x .* y .* v );
    den                     = sqrt( nansum( x .* x .* v ) .* nansum( y .* y .* v) );
    zidx                    = den == 0;
    cc                      = zeros( 1, n );
    cc( ~zidx )             = num( ~zidx ) ./ den( ~zidx );
   
end

return

% EOF