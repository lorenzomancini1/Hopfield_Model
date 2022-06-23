module CH

using LinearAlgebra, Random, Statistics

export init_pattern, overlap, generate_patterns, energy

function init_pattern(N)
    σ = rand(1:5, N)
    return σ
end

function overlap(σ1::AbstractVector, σ2::AbstractVector)
    return σ1 ⋅ σ2
end

function generate_patterns(M, N)
    ξ = rand(1:5, N, M)
    return ξ
end

function energy(σ::AbstractVector, ξ::AbstractMatrix, β, γ)
    M = size(ξ, 2)
    N = length(σ)
    
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

end