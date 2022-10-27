using Random, Statistics, LinearAlgebra
using Plots
using BenchmarkTools
using Test

function generate_hopfield_sample(N, α, zerodiag=true)
    M = round(Int, N*α)
    ξ = rand([-1,1], N, M)
    J = (ξ * ξ') ./ N
    if zerodiag
        J[diagind(J)] .= 0
    end
    return ξ, J
end

rand_config(N::Int) = rand([-1,1], N)

"""
Flip each spin of `σ` with probability `p`.
"""
function perturb_config(σ::AbstractVector, p)
    @assert 0 <= p <= 1
    σnew = copy(σ)
    for i in 1:length(σnew)
        if rand() < p
            flip!(σnew, i)
        end 
    end
    return σnew
end

function energy(J::AbstractMatrix, σ::AbstractVector)
    return -(σ' * J * σ) / 2
end

function staggered_mag(ξ::AbstractVector, σ::AbstractVector)
    return ξ ⋅ σ / length(σ)
end

function staggered_mags(ξ::AbstractMatrix, σ::AbstractVector)
    return ξ' * σ ./ length(σ)
end

function is_local_minimum(J, σ)
    h = J * σ .- diag(J) .* σ
    return σ == sign.(h) 
end

flip!(σ, i::Int) = σ[i] = -σ[i]

function delta_energy(J, σ, i::Int)
    hi = @views J[:, i] ⋅ σ  - J[i, i] * σ[i]  # this dot product dominates the compute time of the whole simulation
    return 2hi * σ[i]
end

grad_ene(J, σ) = -J * σ 

binarize(x) = sign.(x)
binarize!(σ, x) = σ .= sign.(x)

function one_binarynet_step!(J, x, η)
    σ = binarize(x)
    x .+= η .* (J * σ)
    clamp!(x, -1, 1)
end

_round(x) = round(x, digits=4)
_round(x::NamedTuple) = map(_round, x)

function run_binarynet(J::AbstractMatrix;  
            x0::AbstractVector = randn(size(J, 1)), 
            infotime=10, # Set to 0 for no printing
            nsteps=1000,
            η = 0.1,
            ξ0 = nothing, # a reference configuration,  used just during logging
            # earlystop=true, # Stop if fliprate is equal or less than this threshold.
            )
    
    N = size(J, 1)
    @assert length(x0) == N
    x = copy(x0)

    function report(t)
        σ = binarize(x)
        Ebin = energy(J, σ) / N
        Ereal = energy(J, x) / N
        absx = sum(abs, x) / N
        res = (; t, Ebin, Ereal, absx)
        if ξ0 !== nothing
            m0 = staggered_mag(ξ0, σ)
            res = (; res..., m0)
        end
        @info res |> _round
    end

    infotime > 0 && report(0)
    for t in 1:nsteps
        one_binarynet_step!(J, x, η)
        if infotime > 0 && t % infotime == 0
            report(t)
        end
        # earlystop && break
    end
    return binarize(x), x
end


# Random.seed!(17)
N = 1000  
α = 0.1
ξ, J = generate_hopfield_sample(N, α)

scale0 = 0.01 # setting a scale < 1 seems to improve retrieval (although scale0=1 also works quite well)
p = 0.3 # perturb probability
σ0 = perturb_config(ξ[:,1], p)
# σ0 = randn([-1,1], N)
x0 = scale0 .* σ0
σ, x = run_binarynet(J; x0, nsteps=1000, η=0.02, ξ0=ξ[:,1]);

m = staggered_mags(ξ, σ)
m[1]

histogram(m, bins=20, xlims=(-1,1))
