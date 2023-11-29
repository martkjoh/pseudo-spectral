include("../../integrators/etd2D.jl");


N = 512
L = N
dt = 2e-2
D = 1e-5


M = Int(1e3)
frames = 500;

αs = [.2, .3, .4, .6]
vs = [0., .6, .8]
n = size(αs)[1]*size(vs)[1]
av = [(α, v) for α in αs for v in vs];
av = [(i, av[i][1], av[i][2]) for i in 1:n]

φt0 = fill(0., (frames, 2, N, N))
par0  = Par(N, L, dt, 0, 0, 0)
result = fill((φt0, par0), n);

for (i, α, v) in av
    par = Par(N, L, dt, α, v, D)
    @time φt = simulate(par, M, frames);
    result[i] = (φt, par)
end;


for (φt, par) in result
    animate_hm(φt, par);
end  