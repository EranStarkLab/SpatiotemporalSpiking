% MIXMAT            mix matrix elements.
%
%                   Y = MIXMAT(X,DIM,MODE)
%
%                   matrix elements are mixed in dimension dim (1 = cols).
%                   if is zero or left empty, all elements are mixed.
%                   mode is 1 (w/o replacement) or 2 (with).
%                   Y is same size as X.
%
%                   a basis for BS.

% 27-mar-03 ES

% last update
% September 2019

function [ y, indx ]            = mixmat( x, dim, mode )

global resetClock
if isempty( resetClock )
    resetClock                  = 1;
end
if resetClock
    rng( round( rand( 1 ) * sum( 100 * clock ) ), 'v4' )
    resetClock                  = 0;
end

nargs                           = nargin;
if nargs < 1
    error( '1 argument')
end
if nargs < 2 || isempty( dim )
    dim                         = 1; 
end
if nargs < 3 || isempty( mode )
    mode                        = 1; 
end

switch dim
    case 0
        isx                     = size( x ); 
        x                       = x( : ); 
        cdim                    = 1;
    case 1
        cdim                    = dim;
    case 2
        x                       = x'; 
        cdim                    = ( ~( dim - 1 ) ) + 1;
end

sx                              = size( x );
if ~prod( sx )
    y                           = zeros( sx );
    return
end
ridx                            = 0 : sx( 1 ) : ( sx( 2 ) - 1 ) * sx( 1 );

if mode == 1                                            % without replacement
    [ ~, p ]                    = sort( rand( sx ), cdim );
elseif mode == 2                                        % with replacement
    p                           = ceil( rand( sx ) * sx( 1 ) );
end

indx                            = ridx( ones( sx( 1 ), 1 ), : ) + p;

switch dim
    case 0
        y                       = reshape( x( indx ), isx );
    case 1
        y                       = x( indx );
    case 2
        y                       = x( indx )';
end

return

% EOF
