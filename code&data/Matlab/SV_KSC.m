%  KSC algorithm for basic stochastic volatility model
%  Example from the simulated data

clear; 
clc;
nloop=22000;
burnin=2000;
T=2000;
eps=normrnd(0,1,T,1);
sigmav=0.5;
v=normrnd(0,sigmav,T,1);
delta=0.95;
alpha=0.2;
h0=0;
h(1)=alpha+delta*h0+v(1);
y=zeros(T,1);
y(1)=exp(0.5*h(1))*eps(1);
for i=2:T
    h(i)=alpha+delta*(h(i-1)-alpha)+v(i);
    y(i)=exp(0.5*h(i))*eps(i);
end

% prior
phih0 = 0; invVphih = 1/10;
muh0 = 0; invVmuh = 1/100;
nuh = 1; Sh = 0.005;

disp('Starting MCMC.... ');
disp(' ' );
start_time = clock;    
    
% initialize the Markov chain
sigh2 = .05;
phih = .95;
muh = 1;
h = log(var(y)*.8)*ones(T,1);

% initialize for storage
store_theta = zeros(nloop - burnin,3); % [mu muh phih sigh2]
store_exph = zeros(nloop - burnin,T);  % store exp(h_t/2)

% compute a few things outside the loop
newnuh = T/2 + nuh;
rand('state', sum(100*clock) ); randn('state', sum(200*clock) );

for loop = 1:nloop        
        % sample h, muh, phih, sigh2
    Ystar = log(y.^2 + .0001);
    [h,muh,phih,sigh2] = SVAR(Ystar,h,muh,phih,sigh2,[muh0 invVmuh ...
        phih0 invVphih nuh Sh]);    
    if ( mod( loop, 2000 ) ==0 )
        disp(  [ num2str( loop ) ' loops... ' ] )
    end    
    if loop>burnin
        i = loop-burnin;
        store_exph(i,:) = exp(h/2)'; 
        store_theta(i,:) = [muh phih sigh2];
    end    
end

disp( ['MCMC takes '  num2str( etime( clock, start_time) ) ' seconds' ] );
disp(' ' );

thetahat = mean(store_theta)';
exphhat = mean(store_exph)'; 
exphlb = quantile(store_exph,.05)';
exphub = quantile(store_exph,.95)';
tid = linspace(1,2000,T)';
figure; plot(tid, [exphhat exphlb exphub]);
box off; xlim([1 2000]);