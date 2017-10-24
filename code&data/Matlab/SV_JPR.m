%  JPR algorithm for basic stochastic volatility model
%  Example from the simulated data

clear;tic
% data
nsim=22000;
burnin=2000;
T=2000;
eps=normrnd(0,1,T,1);
sigmav=0.5;
v=normrnd(0,sigmav,T,1);
delta=0.95;
alpha=-0.2;
logh0=0;
logh(1)=alpha+delta*logh0+v(1);
y=zeros(T,1);
y(1)=sqrt(exp(logh(1)))*eps(1);
for i=2:T
    logh(i)=alpha+delta*logh(i-1)+v(i);
    y(i)=sqrt(exp(logh(i)))*eps(i);
end

% prior
beta0=zeros(2,1);
iVbeta=[1/100,0;0,1/10];
v0=1; s0=0.005;

% initialize the markov chain
sigv2=0.05;
h=unifrnd(0,1,T,1);

% initial for storage
store_theta=zeros(nsim,3);    % [beta,sigv2]
store_h=zeros(nsim,T);

for isim=1:nsim+burnin
    h_res=[ones(T-1,1) log(h(1:T-1))];
    h_d=log(h(2:T));
    
    % sample beta
    Dbeta=(iVbeta+h_res'*h_res/sigv2)\speye(2);
    beta_hat=Dbeta*(iVbeta*beta0+h_res'*h_d/sigv2);
    C=chol(Dbeta,'lower');
    beta=beta_hat+C*randn(2,1);
    
    % sample sigv2
    sigv2=1/gamrnd(v0+T/2,1/(s0+0.5*(h_d-h_res*beta)'*(h_d-h_res*beta)));
    
    % sample h
    h=IMH(y,[beta;sigv2],h);
    
    if (mod(isim, 5000) == 0)
        disp([num2str(isim) ' loops... ']);
    end
    if isim > burnin
        isave = isim - burnin;
        store_h(isave,:) = sqrt(h)'; 
        store_theta(isave,:) = [beta(1) beta(2) sigv2];
    end
end

theta_hat = mean(store_theta)';
sqrth_hat= mean(store_h)'; 
sqrthlb = quantile(store_h,.05)';
sqrthub = quantile(store_h,.95)';
tid = linspace(1,2000,T)';
figure; plot(tid, [sqrth_hat sqrthlb sqrthub]);
box off; xlim([1 2000]);
toc 