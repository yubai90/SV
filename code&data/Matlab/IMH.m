function h = IMH(y,para,h)
% Draw the state variable h using an independent Metropolis-Hastings
% algorithm 
% Volatility dynamics is AR(1)
% Reference: JPR(1994)

T=length(y);
% parameters
alpha=para(1);
delta=para(2);
sigh2=para(3);

% draw states at time 0
t=0;
mibar=-1; sigmabar=10;                           % prior for period 0 volatility
ss = sigmabar*sigh2/(sigh2 + sigmabar*delta^2);       % conditional variance
mu = ss*(mibar/sigmabar + (log(h(t+1))-alpha)/sigh2);         % conditional mean
h0=exp(mu + (ss^.5)*randn(1,1));

% draw states from time 1 to time T
for t=1:T
    % mean and the variance of the proposal density, see Cogley and Sargent(2004)
    if t==1
        mu=(alpha*(1-delta)+(log(h(t+1))+log(h0))*delta)/(1+delta^2);
        ss=sigh2/(1+delta^2);
    elseif t>1 && t<T
        mu=(alpha*(1-delta)+(log(h(t+1))+log(h(t-1)))*delta)/(1+delta^2);
        ss=sigh2/(1+delta^2);
    elseif t==T
        mu = alpha+delta*log(h(t-1));
        ss = sigh2;
    end
    
    % draw the candidate
    h_cand = exp(mu + (ss^.5)*randn);
    
    % acceptance and reject probability
    lp1 = -0.5*log(h_cand)   - (y(t)^2)/(2*h_cand);     % numerator
    lp0 = -0.5*log(h(t)) - (y(t)^2)/(2*h(t));           % denominator
    
    % decide whether to accept the draw
    if min(1,exp(lp1 - lp0))>=rand;
        h(t)= h_cand; 
    end
    
end
        
end
