module MHB

using Random, Statistics, LinearAlgebra
using KahanSummation

export init_pattern, overlap, generate_patterns, perturb, energy, energy_variation, metropolis, monte_carlo

function init_pattern(N)
    σ = rand([-1, 1], N)
    return σ
end


function overlap(σ1::AbstractVector, σ2::AbstractVector)
    return σ1 ⋅ σ2
end


function generate_patterns(M, N)
    ξ = rand([-1,1], N, M)
    return ξ
end


function perturb(σ::AbstractVector, p)
    N = length(σ)
    σ_new = copy(σ)
    for i in 1:N
        if rand() < p
            σ_new[i] *= -1
        end
    end
    return σ_new
end

function logsumexp(x::AbstractVector)
    imax = argmax(x)
    xmax = x[imax]
    ws = exp.(x .- xmax)
    ws[imax] = 0 # in order to se log1p later
    return xmax + log1p(sum_kbn(ws))
end

function energy(σ::AbstractVector, ξ::AbstractMatrix, λ = 1)
    qs = λ .* vec(σ' * ξ)
    lse = logsumexp(qs)
    return - lse / λ
end

function energy_variation(σ, ξ, i, λ = 1)
    density = exp.(sum(λ .* (ξ .* σ), dims = 1))
    variation = exp.(ξ[i,:] .* (-2 * λ * σ[i]))
    return - log( (density ⋅ variation) / sum(density) )/λ
end


function metropolis!(σ::AbstractVector, ξ::AbstractMatrix, β = 10, λ = 1)
    N, M = size(ξ)
    fliprate = 0
    E = energy(σ, ξ, λ)
    for i in randperm(N)
        σ[i] *= -1
        Enew = energy(σ, ξ, λ)
        ΔE = Enew - E
        if rand() < exp( - β * ΔE)
            fliprate += 1
            E = Enew
        else
            σ[i] *= -1
        end
    end
    
    return σ, fliprate / N
end

function monte_carlo(σ0::AbstractVector, ξ::AbstractMatrix;
                    nsweeps = 100, earlystop = 0, β = 10, λ = 1)

    σ = deepcopy(σ0)
    for sweep in 1:nsweeps
        σ, fliprate = metropolis!(σ, ξ, β, λ)
        fliprate <= earlystop && break
    end
    return σ
end


end # end module