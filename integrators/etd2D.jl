using FFTW
using AbstractFFTs
using Random
using ProgressMeter
using Plots
using BenchmarkTools
using Base.Threads


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
    x::Tuple{Vector{Float64}, Matrix{Float64}}
    k::Tuple{Vector{Float64}, Matrix{Float64}}
    k²::Matrix{Float64}
    f::Tuple{Matrix{Float64}, Matrix{Float64}}
    plan
    g
    AA!
    function Tools(par)
        N, dt, dx, dk, α = par.N, par.dt, par.dx, par.dk, par.α

        x1 = dx .* collect(0:N-1)
        x2 = dx .* collect(0:N-1)'

        k1 = rfftfreq(N , N * dk)
        k2 = fftfreq(N, N * dk)'
        k² = @. (k1^2 + k2^2);

        c = - k².^2
        cdt = c .* dt
        f1 = exp.(cdt)
        f2 = dt * ( exp.(cdt) .- 1 ) ./ cdt
        f2[1, 1] = 0.

        F = plan_rfft(x1 .+ x2)
        B = plan_irfft(im*k², N)

        absk = sqrt.(k²)
        kmax = maximum(sqrt.(absk))
        twothirds = absk .>= kmax * 2/3
        function antialias!(F)
                F[twothirds] .= 0.
        end

        function g(φ1, φ2)
            urφ² = @. 1 - (φ1^2 + φ2^2)
            a1 = @. urφ² * φ1 + α * φ2
            a2 = @. urφ² * φ2 - α * φ1
            b1 = k² .* (F * a1)
            b2 = k² .* (F * a2)
            return b1, b2
        end

        x = (x1, x2)
        k = (k1, k2)
        f = (f1, f2)
        plan = (F, B)
        
        new(x, k, k², f, plan, g, antialias!)
    end
end

function etd!(par, tools, Fφ, φ)
    N, D = par.N, par.D

    k² = tools.k²
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
    
    # ξ1 = rand(N, N) .* D
    # ξ2 = rand(N, N) .* D
    # kξ1 = k² .* (F * ξ1)
    # kξ2 = k² .* (F * ξ2)
    # Fφ1 .+= kξ1
    # Fφ2 .+= kξ2

    φ1 .= B*Fφ1
    φ2 .= B*Fφ2
end

function init_cos(par, tools; A=1e-1, Δ=2., n=1.)
    x1, x2 = tools.x
    N, dk = par.N, par.dk
    δ1 = @. A * cos(dk*n*x1) * cos(dk*n*x2)
    δ2 = @. A * cos(dk*n*x1) * cos(dk*n*(x2 + Δ))

    v = par.v
    φ1 = v * ones(N, N) + δ1
    φ2 = zeros(N, N) + δ2

    return φ1, φ2
end


function init_random(par, tools;)
    v, N = par.v, par.N
    φ1 = randn(N, N)
    φ2 = randn(N, N)

    av1 = sum(φ1) / N^2
    av2 = sum(φ2) / N^2

    @. φ1 += (v - av1)
    @. φ2 += -av2

    return φ1, φ2
end


function run(par, tools, M, frames; init=init_random)
    N = par.N
    φ1, φ2 = init(par, tools)
    F, B = tools.plan;
    Fφ1, Fφ2 = F*φ1, F*φ2;

    φt = Array{Float64}(undef, (frames, 2, N, N))
    φt[1, 1, :, :] .= φ1
    φt[1, 2, :, :] .= φ2

    n = M ÷ frames
    @showprogress 1 "Simulating:" for i in 2:frames
        for j in 1:n
            etd!(par, tools, (Fφ1, Fφ2), (φ1, φ2))
        end
        @assert !any(isnan.(Fφ1))
        φt[i, 1, :, :] .= φ1
        φt[i, 2, :, :] .= φ2
    end
    return φt
end

function simulate(par, M, frames)
    tools = Tools(par);
    φt = run(par, tools, M, frames)
    return φt
end;

function animate(φt, par; skip=1)
    v,α = par.v, par.α
    name = "NRCH_α=$(α)_v=$(v)"
    frames = size(φt)[1]
    x1, x2 = par.x
    anim = @animate for i in 1:frames
        plot()
        p1 = plot(x1, x2, φt[i, 1, :, :]; st=:surface)
        p2 = plot(x1, x2, φt[i, 2, :, :]; st=:surface)
        plot(p1, p2)
        plot!(;zrange=(-1.1, 1.1), clim=(-1.1, 1.1))
        plot!(size=(1600, 800))
    end every skip;
    gif(anim, name*".mp4", fps = 10)
end

function animate_hm(φt, par; skip=1)
    v,α = par.v, par.α
    name = "vid/NRCH_HM_α=$(α)_v=$(v)"
    frames = size(φt)[1]
    p = Progress(frames, 1, "Animating :") 
    anim = @animate for i in 1:frames
        plot()
        p1 = heatmap(φt[i, 1, :, :]; c=:viridis)
        p2 = heatmap(φt[i, 2, :, :]; c=:viridis)
        plot(p1, p2)
        plot!(;clim=(-1.1, 1.1), size=(1600, 800))
        next!(p)
    end every skip;
    gif(anim, name*".mp4", fps = 10)
end




