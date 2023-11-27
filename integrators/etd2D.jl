using FFTW
using AbstractFFTs

struct Par
    N::Int
    L::Float64
    α::Float64
    v::Float64
    x
    k
    k²
    f
    plan
    AA!
end

function setup(N, L, dt, α, v)
    dx = L/N
    dk = 2π/L

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

    absk = sqrt.(k²)
    kmax = maximum(sqrt.(absk))
    twothirds = absk .>= kmax * 2/3
    function antialias!( F )
            F[twothirds] .= 0.
    end
    F = plan_rfft(x1 .+ x2)
    B = plan_irfft(im*absk, N)

    x = (x1, x2)
    k = (k1, k2)
    f = (f1, f2)
    plan = (F, B)
    par = Par(N, L, α, v, x, k, k², f, plan, antialias!)
    return par
end

function get_g(par)
    F, α, k² = par.plan[1], par.α, par.k²
    function g(φ1, φ2)
        urφ² = @. 1 - (φ1^2 + φ2^2)
        a1 = @. urφ² * φ1 + α * φ2
        a2 = @. urφ² * φ2 - α * φ1
        b1 = k² .* (F * a1)
        b2 = k² .* (F * a2)
        return b1, b2
    end
end

function etd!(par, g, Fφ, φ)
    F, B = par.plan
    f1, f2 = par.f
    Fφ1, Fφ2 = Fφ
    φ1, φ2 = φ
    g1, g2 = g(φ1, φ2)
    Fφ1 .= f1 .* Fφ1 + f2 .* g1
    Fφ2 .= f1 .* Fφ2 + f2 .* g2
    par.AA!(Fφ1)
    par.AA!(Fφ2)
    φ1 .= B*Fφ1
    φ2 .= B*Fφ2
end

function check(i, j, n, M, Fφ)
    t = i*n + j
    if t%(M÷10) == 0
        print(100*t/M,"% \n")
        flush(stdout)
        @assert !any(isnan.(Fφ))
    end
end

function run(par, g, M, frames)
    A = 1e-1
    Δ = .2
    x1, x2 = par.x
    δ1 = @. A * cos(dk*x1) * cos(dk*x2)
    δ2 = @. A * cos(dk*x1) * cos(dk*(x2 + Δ))

    v = par.v
    φ1 = v * ones(N, N) + δ1
    φ2 = zeros(N, N) + δ2
    F, B = par.plan;
    Fφ1, Fφ2 = F*φ1, F*φ2;

    φt = Array{Float64}(undef, (frames, 2, N, N))
    φt[1, 1, :, :] .= φ1
    φt[1, 2, :, :] .= φ2

    n = M ÷ frames
    for i in 2:frames
        for j in 1:n
            etd!(par, g, (Fφ1, Fφ2), (φ1, φ2))
            check(i, j, n, M, Fφ1)
        end
        φt[i, 1, :, :] .= φ1
        φt[i, 2, :, :] .= φ2
    end
    return φt
end

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
    name = "NRCH_HM_α=$(α)_v=$(v)"
    frames = size(φt)[1]
    anim = @animate for i in 1:frames
        plot()
        p1 = heatmap(φt[i, 1, :, :]; c=:viridis)
        p2 = heatmap(φt[i, 2, :, :]; c=:viridis)
        plot(p1, p2)
        plot!(;clim=(-1.1, 1.1), size=(1600, 800))
    end every skip;
    gif(anim, name*".mp4", fps = 10)
end
