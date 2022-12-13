using Hopfield_Model: MHC
using LinearAlgebra, Random, Statistics
using DataFrames
using Plots, StatsPlots

N = 40
α = 0.2
λ = 0.1

M = round(Int, exp(α * N))
ξ = MHC.generate_patterns(N, M)
σ0 = ξ[:,1]
MHC.overlap(σ0, σ0) # 1.0

σ, res =  MHC.gradient_descent(σ0, ξ; λ, η=0.1, nsteps=100)
df = DataFrame(res)
@df df scatter(:t, :E, label="E", xlabel="t")
@df df scatter!(:t, :Δ0, label="Δ₀")
