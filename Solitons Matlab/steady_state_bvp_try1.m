L = 1;

xmesh = linspace(-L/2,L/2,100);
solinit = bvpinit(xmesh, @(x)guess(x,L));
epsilon = 0.1; D = 0.6; Q=5; V = 1; alpha = 0.95;

options = bvpset('RelTol',1e-4,'Stats','on');
sol = bvp5c(@(x,y)odefun2(x,y, epsilon, D,Q,V,alpha), @bcfcn2, solinit, options);
% cubic solve
steady_state = fsolve(@(eta)qubic_eq(eta,V,alpha,Q), 0);

plot(sol.x, sol.y(1,:)); hold on; plot(sol.x, steady_state.*ones(length(sol.x))); hold off
function g = guess(x,L)
    g = [sin(2.*pi.*x./L)+1.4
     cos(2.*pi.*x./L)
     -sin(2.*pi.*x./L)];
end

function qubic =  qubic_eq(eta, V,alpha,Q)
qubic = eta.^3./3+V./(alpha.*2).*eta.^2+V.*eta+V.*alpha./2-Q;

end

function res = bcfcn2(ya,yb)
res = [ya(1)-yb(1)
       ya(2)-yb(2)
       ya(3)-yb(3)];
end

function [dydt] = odefun2(x,y, epsilon, D,Q,V,alpha)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

dydt = zeros(3,1); 
dydt(1) = y(2); %y'
dydt(2) = y(3); %y''
dydt(3) = 1./(epsilon.^2).*(3.*Q./y(1).^3-1-D.*y(2)-3.*V./alpha./y(1).^3./2.*(1+y(1)./alpha).^2);
end