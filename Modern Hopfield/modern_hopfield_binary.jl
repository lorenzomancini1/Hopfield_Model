module MH_binary

using Random, Statistics, LinearAlgebra

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


function energy(σ::AbstractVector, ξ::AbstractMatrix, λ = 1)
    M = size(ξ, 2)
    se = 0
    for μ in 1:M
        se += exp(λ * overlap(σ, ξ[:, μ]) )
    end
    E = - log(se)/λ
    return E
end

function energy_variation(σ, ξ, i, λ = 1)
    density = exp.(sum(λ .* (ξ .* σ), dims = 1))
    variation = exp.(ξ[i,:] .* (-2 * λ * σ[i]))
    return - log( (density ⋅ variation) / sum(density) )/λ
end


function metropolis(σ::AbstractVector, ξ::AbstractMatrix, β = 10, λ = 1)
    
    N = length(σ)
    M = size(ξ, 2)
    fliprate = 0
    
    for n in 1:N
        i = rand(1:N)
            
        ΔE = energy_variation(σ, ξ, i, λ)
        
        if ΔE < 0 || rand() < exp( - β * ΔE)
            σ[i] *= -1
            fliprate += 1
        end
    end
    
    return σ, fliprate/N
end

function monte_carlo(σ::AbstractVector, ξ::AbstractMatrix, nsweeps, earlystop = 0, β = 10, λ = 1)
    
    for sweep in 1:nsweeps
        σ, fliprate = metropolis(σ, ξ, β, λ)
        fliprate <= earlystop && break
    end
    return σ
end

end