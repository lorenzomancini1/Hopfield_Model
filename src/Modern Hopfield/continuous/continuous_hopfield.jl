module CH

using LinearAlgebra, Random, Statistics, Distributions

export init_pattern, overlap, generate_patterns, energy

function init_pattern(N)
    #σ = rand(1:5, N) #cambiare
    σ = randn(N)
    return σ
end

function distance(σ1::AbstractVector, σ2::AbstractVector)
    return norm(σ1 - σ2)
end

function generate_patterns(M, N)
    #ξ = 1. + 4 * rand(N, M)
    ξ = randn(N, M)
    return ξ
end

function perturb(σ::AbstractVector, Δ::Float64)
    N = length(σ)
    σ_new = copy(σ)
    noise = rand(Normal(0, Δ), N)
    return σ_new + noise
end

function energy(σ::AbstractVector, ξ::AbstractMatrix; β = 10, γ=1)
    M = size(ξ, 2)
    #N = length(σ)
    
    max_norm_sq = findmax(sum(ξ.^2, dims = 1))[1]
    #print(max_norm_sq)
    se = 0
    
    for μ in 1:M
        se += exp(β * overlap(σ, ξ[:, μ]))
    end
    lse = log(se)/β
    energy = -lse + 0.5*γ*sum(σ.^2) + log(M)/β + 0.5*max_norm_sq
    
    if energy >= 10^4
        return 10^4
    else
        return energy
    end
end

function softmax(x)
    m = maximum(x)
    return exp.(x .- m) ./ sum(exp.(x)) 
end

function update(σ, ξ; β = 1, nsweeps = 1)
    σ_rec = copy(σ)
    for _ in 1:nsweeps
        σ_rec = ξ * softmax(β * (ξ' * σ_rec))
    end
    return σ_rec
end

end