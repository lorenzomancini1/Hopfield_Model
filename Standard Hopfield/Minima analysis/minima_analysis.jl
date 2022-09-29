include("../standard_hopfield.jl")
using Statistics, LinearAlgebra, Plots
using DelimitedFiles, Random
using ProfileView
using BenchmarkTools

function hist_overlaps(;
    α = 0.05, N = 1000, nbins = 60, nsweeps = 100,
     earlystop = 0, β = 10, annealing = false,
    show = false, save = false)

M = round(Int, N*α)
ξ = SH.generate_patterns(M, N)
J = SH.store(ξ)
    # take a random configuration
σ = rand([-1,1], N)
σ_new = SH.monte_carlo(J, σ; nsweeps = nsweeps, earlystop = earlystop, β = β, annealing = annealing)

overlaps = (σ_new' * ξ) ./ N

if show
    fig = histogram(overlaps', nbins = nbins, label = "N = $N, α = $α", xlabel = "overlap")
    display(fig)
end

if save
    path = "overlaps"

    if isdir(path)
        io = open(path*"/N"*"$N"*".txt", "w") do io
            writedlm(io, [overlaps])
        end
    else
        mkdir(path)
        io = open(path*"/N"*"$N"*".txt", "w") do io
            writedlm(io, [overlaps])
        end
    end
end

return overlaps
end