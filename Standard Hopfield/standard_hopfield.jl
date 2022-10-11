module SH

using Random, Statistics, LinearAlgebra
using LoopVectorization, Tullio

export init_pattern, overlap, generate_patterns, perturb, energy, energy_variation, store, metropolis, monte_carlo

function init_pattern(N::Int)
    σ = rand([-1, 1], N)
    return σ
end

function overlap(σ1::AbstractVector, σ2::AbstractVector)
    return σ1 ⋅ σ2 / length(σ1)
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

function store(ξ)
    N = size(ξ, 1)
    J = ξ * ξ'
    J[diagind(J)] .= 0
    return J ./ N
end

function energy(J::AbstractMatrix, σ::AbstractVector)
    return -(σ' * J * σ) / 2
end

#function energy_variation(J::AbstractMatrix, σ::AbstractVector, i::Int)
#    return @views 2 * σ[i] * (J[:, i] ⋅ σ)
#end

# ~30% faster
function energy_variation(J::AbstractMatrix, σ::AbstractVector, i::Int)
    s = 0.0
    @tturbo for j in eachindex(σ)
        s += J[j, i] * σ[j]
    end
    return 2 * σ[i] * s
end
# approx. the same as tturbo version
# function energy_variation(J::AbstractMatrix, σ::AbstractVector, i::Int)
#     Ji = @view J[:,i]
#     @tullio s := Ji[j] * σ[j] threads=true
#     return 2 * σ[i] * s
# end

function metropolis(J, σ, β)
    N = length(σ)
    
    fliprate = 0
    is = randperm(N)
    for i in is
        ΔE = energy_variation(J, σ, i)
        
        if (ΔE < 0) || rand() < exp(-β*ΔE)
            σ[i] *= -1
            fliprate += 1
        end
    end
    return σ, fliprate/N
end

function monte_carlo(J::AbstractMatrix, σ::AbstractVector; 
        nsweeps = 100, earlystop = 0, β = 10, annealing = false)
    
    σ_rec = copy(σ)

    if annealing
        Tf  = 1 / β 
        T = range(2, Tf, length=nsweeps)
        β_n = 1 ./ T
        for β in β_n
            σ_rec, fliprate = metropolis(J, σ_rec, β)
            fliprate <= earlystop && break
        end
    else
        for sweep in 1:nsweeps
            σ_rec, fliprate = metropolis(J, σ_rec, β)
            fliprate <= earlystop && break
        end
    end
    return σ_rec
end

end