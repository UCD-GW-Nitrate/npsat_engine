function WellURF = fitURFs( WellURF )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
ft = fittype( '(1/(x*b*sqrt(2*pi)))*exp((-(log(x)-a)^2)/(2*b^2))', 'independent', 'x', 'dependent', 'y' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.StartPoint = [0.499112701571756 0.336174051321482];

for ii = 1:size(WellURF,1)
    X = 1:length(WellURF(ii,1).URF);
    [xData, yData] = prepareCurveData( X, WellURF(ii,1).URF );
    [fitresult, gof] = fit( xData, yData, ft, opts );
    ab = coeffvalues(fitresult);
    WellURF(ii,1).m = ab(1);
    WellURF(ii,1).s = ab(2);
end

