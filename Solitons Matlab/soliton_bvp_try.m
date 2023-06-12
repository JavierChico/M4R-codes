L = 2;
xmesh = linspace(-L/2,L/2,50);
solinit = bvpinit(xmesh, @guess, 1);


options = bvpset('RelTol',1e-3,'Stats','on') ;
sol = bvp5c(@odefun1, @bcfcn1, solinit, options);

plot(sol.x, sol.y(1,:))
function g = guess(x)
    L = 2;
    g = [sin(2.*pi.*x./L)+2
     cos(2.*pi.*x./L)
     -sin(2.*pi.*x./L)
     -cos(2.*pi.*x./L)
     sin(2.*pi.*x./L)];
end

function res = bcfcn1(ya,yb,c)
L = 2;
res = [ya(1)-yb(1)
       ya(2)-yb(2)
       ya(3)-yb(3)
       ya(4)-yb(4)
       ya(5)
       yb(5)-L];
end

function [dydt] = odefun1(x,y,c)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
epsilon = 0.1; D = 0.6;
dydt = zeros(5,1); 
dydt(1) = y(2); %y'
dydt(2) = y(3); %y''
dydt(3) = y(4); %y'''
dydt(4) = 3./(epsilon.^2.*y(1).^3).*(c.*y(2)-(1+D).*y(1).^2.*y(2)-D./3.*y(1).^3.*y(3)-epsilon.^2.*y(1).^2.*y(4));
dydt(5) = y(1).^2;
end