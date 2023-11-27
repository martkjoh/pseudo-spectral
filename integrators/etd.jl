using FFTW
using AbstractFFTs

struct Par
    N::Int
    L::Float64
    α::Float64
    x::Vector{Float64}
    k::Vector{Complex}
    k1::Vector{Complex}
    k2::Vector{Complex}
    F
    B
end

function setup(N, L, dt, α)
    dx = L/N
    dk = 2*π / L

    x = dx .* collect(0:N-1)
    k = dk .* collect(0:N÷2)

    c = -k.^4
    cdt = c .* dt
    k1 = exp.(cdt)
    k2 = dt * ( exp.(cdt) .- 1 ) ./ cdt
    k2[1] = 1.

    F = plan_rfft(x)
    B = plan_irfft(im * k, N)

    par = Par(N, L, α, x, k, k1, k2, F, B)

    return par
end

function get_g(par)
    F, α, k = par.F, par.α, par.k
    k² = k.^2
    function g(φ1, φ2)
        urφ² = 1 .- (φ1.^2 .+ φ2.^2)
        g1 = k² .* (F * (urφ².*φ1 .+ α .* φ2 ))
        g2 = k² .* (F * (urφ².*φ2 .- α .* φ1 ))
        return g1, g2
    end

    
end

function antialias!(F)
    F[end-(N÷6)+1:end] .= 0.
end

function etd!(par, g, Fφ, φ)
    Fφ1, Fφ2 = Fφ
    φ1, φ2 = φ 
    g1, g2 = g(φ1, φ2)
    Fφ1 .= par.k1 .* Fφ1 + par.k2 .* g1
    Fφ2 .= par.k1 .* Fφ2 + par.k2 .* g2
    antialias!(Fφ1)
    antialias!(Fφ2)
    φ1 .= par.B*Fφ1
    φ2 .= par.B*Fφ2
end
