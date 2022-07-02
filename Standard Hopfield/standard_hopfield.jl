module SH

using Random, Statistics, LinearAlgebra

export init_pattern, overlap, generate_patterns, perturb, energy, energy_variation, store, metropolis, monte_carlo

function init_pattern(N::Int)
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

function store(ξ::AbstractMatrix)
    N = size(ξ, 1)
    J = ξ * ξ'
    J[diagind(J)] .= 0
    return J ./ N
end

function energy(J::AbstractMatrix, σ::AbstractVector)
    return -(σ' * J * σ) / 2
end

function energy_variation(J::AbstractMatrix, σ::AbstractVector, i::Int)
    return @views 2 * σ[i] * (J[:, i] ⋅ σ)
end

function metropolis(J, σ, β)
    N = length(σ)
    
    fliprate = 0
    for n in 1:N
        i = rand(1:N)
        ΔE = energy_variation(J, σ, i)
        
        if (ΔE < 0) || rand() < exp(-β*ΔE)
            σ[i] *= -1
            fliprate += 1
        end
    end
    return σ, fliprate/N
end

function monte_carlo(J::AbstractMatrix, σ::AbstractVector; nsweeps = 100, earlystop = 0, β = 10)
    
    σ_rec = copy(σ)
    for sweep in 1:nsweeps
        σ_rec, fliprate = metropolis(J, σ_rec, β)
        fliprate <= earlystop && break
    end
    return σ_rec
end

end