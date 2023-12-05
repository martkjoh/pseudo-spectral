using FFTW
using AbstractFFTs
using Random
using Plots
using ProgressMeter


struct Par
    N::Int
    L::Float64
    dt::Float64
    α::Float64
    v::Float64
    D::Float64
    dk::Float64
    dx::Float64
    function Par(N, L, dt, α, v, D) 
        dx = L/N
        dk = 2π/L
        new(N, L, dt, α, v, D, dk, dx)
    end
end

struct Tools
    x
    k
    k²
    f
    plan
    g
    AA!
    function Tools(par)
        N, dt, dx, dk, α = par.N, par.dt, par.dx, par.dk, par.α

        x = dx .* collect(0:N-1)
        k = dk .* collect(0:N÷2)

        k² = k.^2
        
        c = -k.^4
        cdt = c .* dt
        f1 = exp.(cdt)
        f2 = dt * ( exp.(cdt) .- 1 ) ./ cdt
        f2[1] = 1.
    
        F = plan_rfft(x)
        B = plan_irfft(im * k, N)
 
        absk = sqrt.(k²)
        kmax = maximum(sqrt.(absk))
        twothirds = absk .>= kmax * 2/3
        function antialias!(F)
                F[twothirds] .= 0.
        end

        # function antialias!(F)
        #     F[end-(N÷6)+1:end] .= 0.
        # end


        # function g(φ1, φ2)
        #     urφ² = 1 .- (φ1.^2 .+ φ2.^2)
        #     g1 = k² .* (F * (urφ².*φ1 .+ α .* φ2 ))
        #     g2 = k² .* (F * (urφ².*φ2 .- α .* φ1 ))
        #     return g1, g2
        # end
        
        function g(φ1, φ2)
            urφ1² = @. 1. *(1 - φ1^2)
            urφ2² = @. 1. *(1 - φ2^2)
            g1 = k² .* (F * (urφ1².*φ1 .+ α .* φ2 ))
            g2 = k² .* (F * (urφ2².*φ2 .- α .* φ1 ))
            return g1, g2
        end

        f = (f1, f2)
        plan = (F, B)
        
        new(x, k, k², f, plan, g, antialias!)
    end
end

function etd!(par, tools, Fφ, φ)
    g = tools.g
    AA! = tools.AA!
    F, B = tools.plan
    f1, f2 = tools.f
    
    Fφ1, Fφ2 = Fφ
    φ1, φ2 = φ
    g1, g2 = g(φ1, φ2)

    Fφ1 .= f1 .* Fφ1 + f2 .* g1
    Fφ2 .= f1 .* Fφ2 + f2 .* g2

    AA!(Fφ1)
    AA!(Fφ2)
    
    φ1 .= B*Fφ1
    φ2 .= B*Fφ2
end

function init_wave(par, tools; A=1, n=1.)
    x = tools.x
    N, dk = par.N, par.dk
    δ1 = @. A * cos(dk*n*x)
    δ2 = @. A * sin(dk*n*x)

    v = par.v
    φ1 = v * ones(N) + δ1
    φ2 = zeros(N) + δ2

    return φ1, φ2
end

function init_random(par, tools;)
    v, N = par.v, par.N
    φ1 = randn(N)
    φ2 = randn(N)

    av1 = sum(φ1) / N
    av2 = sum(φ2) / N

    @. φ1 += (v - av1)
    @. φ2 += -av2

    return φ1, φ2
end


function shape(par)
    N = par.N
    b1 = (N ÷ 40)
    b2 = 3 * b1

    n1 = 4 * b1
    n2 = 10 * b1
    return b1, b2, n1, n2
end

function init_square(par, tools;)
    N = par.N
    φ1 = - ones(N)
    φ2 = - ones(N)
    b1, b2, n1, n2 = shape(par)

    @. φ1[b1:b1+n1]  = + 1
    @. φ2[b2:b2+n2]  = + 1

    return φ1, φ2
end


function run(par, tools, M, frames; init=init_wave)
    N = par.N
    φ1, φ2 = init(par, tools)
    F, B = tools.plan;
    Fφ1, Fφ2 = F*φ1, F*φ2;

    φt = Array{Float64}(undef, (frames, 2, N))
    φt[1, 1, :] .= φ1
    φt[1, 2, :] .= φ2

    n = M ÷ frames
    @showprogress 1 "Simulating:" for i in 2:frames
        for j in 1:n
            etd!(par, tools, (Fφ1, Fφ2), (φ1, φ2))
        end
        @assert !any(isnan.(Fφ1))
        φt[i, 1, :] .= φ1
        φt[i, 2, :] .= φ2
    end
    return φt
end

function simulate(par, M, frames;init=init_wave)
    tools = Tools(par);
    φt = run(par, tools, M, frames;init=init)
    return φt, tools
end;


function animate(φt, par, tools)
    x = tools.x
    anim = @animate for i in 1:frames
        plot()
        plot!(x, φt[i, 1, :])
        plot!(x, φt[i, 2, :])
        plot!(;yrange=(-1.1, 1.1))
    end;
    gif(anim, "vid1D/nrch.mp4", fps = 30)
    
end

function plot_wall(φt, par, tools, M)
    x = tools.x
    dx = par.dx
    frames = size(φt)[1]
    t = 1:frames
    walls = zeros(2, frames)
    for k in 1:2
        φ1 = φt[:, k, :]
        for i in t
            s = sign.(φ1[i, :])
            cross = (s  - circshift(s, 1)) .> 1
            crossind = findall(cross)[1]
            x1 = crossind * dx
            x2 = (crossind - 1) * dx
            y1 = φ1[i, crossind]
            y2 = φ1[i, crossind-1]
            x = x2 - (x1 - x2)/(y1 - y2) * y2

            walls[k, i] = x
        end 
    end

    α, L, N, dt, dx = par.α, par.L, par.N, par.dt, par.dx
    dt = M / frames  * dt

    b1, b2, n1, n2 = shape(par)
    S1 = n1 * dx
    S2 = n2 * dx

    v1 = α / S1 * L / (L - S1)
    v2 = α / S2 * L / (L - S2)
    print(v1, '\n')
    print(v2)

    p1 = plot()
    T = dt .* (collect(t) .- 1)
    x01 = walls[1,1]
    x02 = walls[2,1]
    
    T = T
    x1 = x01 .+ v1 .* T
    x2 = x02 .+ v2 .* T

    plot!(T, walls[1,:])
    plot!(T, walls[2,:])
    plot!(T, x1)
    plot!(T, x2)
    plot!(size=(1000, 500))
    
    display(p1)

    p2 = plot()
    plot!(T[1:end-1], (walls[1,2:end] - walls[1,1:end-1]) / dt)
    plot!(T[1:end-1], (walls[2,2:end] - walls[2,1:end-1]) / dt)
    plot!(T, v1*ones(frames))
    plot!(T, v2*ones(frames))
    plot!(size=(1000, 500))

    display(p2)

    
end
    