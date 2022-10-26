using Hopfield_Model
using Test
using Random, Statistics, LinearAlgebra



N = 50
α = 0.3
M = round(Int, exp(α*N))
# Random.seed!(17)
ξ = MHB.generate_patterns(M, N)
# σ0 = MHB.init_pattern(N)
σ0 = ξ[:,1]
@assert size(ξ) == (N, M)
@assert length(σ) == N

σ =  MHB.monte_carlo(σ0, ξ; nsweeps = 10, earlystop = 0, β = 10^2, λ = 0.2)
qs = vec(σ' * ξ) ./ N
qs[1]