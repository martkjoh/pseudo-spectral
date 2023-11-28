include("../../integrators/etd2D.jl");


N = 500
L = N/10
dt = 1e-2
α = .5
v = .0
D = 1e-5


M = Int(5e3)
frames = 100;

par = Par(N, L, dt, α, v, D)
φt = simulate(par, M, frames);
animate_hm(φt, par);

