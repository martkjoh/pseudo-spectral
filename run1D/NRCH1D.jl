include("../integrators/etd1D.jl");

N = 20_000
L = 2000
dt = 1e-1
α = .01
v = 0.5
D = 2e-2
    

M = Int(1e7)
frames = 500;

par = Par(N, L, dt, α, v, D)
φt, tools = simulate(par, M, frames, init=init_square);
animate(φt, par, tools);
plot_wall(φt, par, tools, M)
