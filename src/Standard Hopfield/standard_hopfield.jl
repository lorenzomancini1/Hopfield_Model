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

function generate_corr_patterns(M, N, q)
    ξ = zeros(M, N)' # initialize ξ
    σ = init_pattern(N)
    ξ[:, 1] = σ # generate the first init_pattern
    for m in 2:M
        ξ[:, m] = perturb(σ, (1-q)/2)
    end
    return ξ
end

function store(ξ)
    N = size(ξ, 1)
    J = ξ * ξ'
    J[diagind(J)] .= 0
    return J ./ N
end

function energy( σ::AbstractVector, J::AbstractMatrix)
    return -(σ' * J * σ) / 2
end

#function energy_variation(J::AbstractMatrix, σ::AbstractVector, i::Int)
#    return @views 2 * σ[i] * (J[:, i] ⋅ σ)
#end

# ~30% faster
function energy_variation(σ::AbstractVector, J::AbstractMatrix, i::Int)
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

function metropolis!(σ, J, β)
    N = length(σ)
    
    fliprate = 0
    for i in randperm(N)
        ΔE = energy_variation(σ, J, i)
        
        if (ΔE < 0) || rand() < exp(-β*ΔE)
            σ[i] *= -1
            fliprate += 1
        end
    end
    return σ, fliprate/N
end

function monte_carlo(σ::AbstractVector, J::AbstractMatrix; 
        nsweeps = 100, earlystop = 0, β = 10, annealing = 0)
    
    σ_rec = deepcopy(σ)

    if annealing == 1
        Tf  = 1 / β 
        T = range(2, Tf, length=nsweeps)
        βn = 1 ./ T
        for β in βn
            σ_rec, fliprate = metropolis!(σ_rec, J, β)
            fliprate <= earlystop && break
        end
    elseif annealing == 0
        for _ in 1:nsweeps
            σ_rec, fliprate = metropolis!(σ_rec, J, β)
            fliprate <= earlystop && break
        end
    elseif annealing == -1
        Ti = 1 / β
        T = 10 .^ range( log10(Ti), log10(15), length = nsweeps )
        βn = 1 ./ T
        for β in βn
            σ_rec, fliprate = metropolis!(σ_rec, J, β)
            fliprate <= earlystop && break
        end
    end
    return σ_rec
end

end