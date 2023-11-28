include("../../integrators/etd2D.jl");


N = 100
L = N/10
dt = 1e-2
α = .5
v = .5
D = 1e-5


M = Int(1e2)
frames = 10;

αs = [.2, .3, .4, .5]
vs = [0., .6, .7, .8]
n = size(αs)[1]*size(vs)[1]
av = [(α, v) for α in αs for v in vs];
av = [(i, av[i][1], av[i][2]) for i in 1:n]

φt = fill(0., (frames, 2, N, N))
par  = Par(N, L, dt, 0, 0, 0)
result = fill((φt, par), n);

@threads for (i, α, v) in av
    par = Par(N, L, dt, α, v, D)
    φt = simulate(par, M, frames);
    result[i] = (φt, par)
end;

for (φ, par) in result
    animate_hm(φt, par);
end