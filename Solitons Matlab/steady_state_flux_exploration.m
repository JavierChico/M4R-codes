clear steady_state_vec
Q_vec = linspace(-10,10, 100);
steady_state_vec = zeros(length(Q_vec));
V = -.1; alpha = 0.95;
for j=1:length(Q_vec)
    steady_state_vec(j) = fsolve(@(eta)qubic_eq(eta,V,alpha,Q_vec(j)), 1);
end

plot(Q_vec, steady_state_vec); hold on
plot(Q_vec, (3.*Q_vec).^(1/3)+V.*((3.*Q_vec).^(-1/3)+alpha/2*(3.*Q_vec).^(-2/3)-1/(2.*alpha)))
function qubic =  qubic_eq(eta, V,alpha,Q)
qubic = eta.^3./3+V./(alpha.*2).*eta.^2+V.*eta+V.*alpha./2-Q;

end