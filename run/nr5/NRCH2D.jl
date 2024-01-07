include("../../integrators/etd2D.jl");


N = 16
L = N
dt = 1e-2
α = 0
v = 0.
D = 0


M = Int(1e1)
frames = 10;

par = Par(N, L, dt, α, v, D)
φt = simulate(par, M, frames;init=init_wave);
animate_hm(φt, par);

# tools = Tools(par);
# tools.f[1]