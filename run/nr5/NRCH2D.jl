include("../../integrators/etd2D.jl");


N = 256
L = N
dt = 2e-1
α = .18
v = 0.
D = 1e-1


M = Int(5e4)
frames = 10;

par = Par(N, L, dt, α, v, D)
φt = simulate(par, M, frames;init=init_wave);
animate_hm(φt, par);

