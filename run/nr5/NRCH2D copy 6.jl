include("../../integrators/etd2D.jl");


N = 512
L = N
dt = 2e-2
α = .4
v = .8
D = 0

M = Int(4e5)
frames = 500;

par = Par(N, L, dt, α, v, D)
φt = simulate(par, M, frames);
animate_hm(φt, par);

