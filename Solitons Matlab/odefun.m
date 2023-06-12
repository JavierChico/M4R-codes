function [dydt] = odefun(x,y)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
epsilon = 0.1; c = 1.13; D = 0.6; A =1;
dydt = zeros(3,1); 
dydt(1) = y(2);
dydt(2) = y(3);
dydt(3) = 1/epsilon.^2.*(3.*c./y(1).^2-1-D.*y(2) + A);
end