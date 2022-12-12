include("../standard_hopfield.jl")
using Statistics, LinearAlgebra, Plots
using DelimitedFiles, Random
using ProfileView
using BenchmarkTools

function show_hist(mf, N, α, nbins)
    fig = histogram(mf, nbins = nbins, label = "N = $N, α = $α", xlabel = "overlap")
    display(fig)
end

function save_data(mf, N, α, p, dir)

    folder_p = replace(string(p),"." => "" )
    folder_α = replace(string(α),"." => "" )
    path = dir*"/p"*folder_p*"/alpha"*folder_α

    if isdir(path)
        io = open(path*"/N"*"$N"*".txt", "w") do io
            writedlm(io, [mf])
        end
    else
        mkpath(path)
        io = open(path*"/N"*"$N"*".txt", "w") do io
            writedlm(io, [mf])
        end
    end
end

function hist_overlaps(;
    α = 0.05, N = 1000, nbins = 60, nsweeps = 100,
     earlystop = 0, p = -1, β = 10, annealing = 0,
    show = false, save = false, savedir = "julia_data")

M = round(Int, N*α)

    # take a random configuration
if p < 0
    ξ = SH.generate_patterns(M, N)
    J = SH.store(ξ)
    σ = rand([-1,1], N)
    σ_new = SH.monte_carlo(σ, J; nsweeps = nsweeps, earlystop = earlystop, β = β, annealing = annealing)
    mf = ((σ_new' * ξ) ./ N)'
    show && show_hist(mf, N, α, nbins)
    save && save_data(mf, N, α, p, savedir)
else
    t = 500
    mf = zeros(t)
    for i in 1:t
        ξ = SH.generate_patterns(M, N)
        J = SH.store(ξ)
        k = rand(1:M)
        σ = ξ[:, k]
        σ_new = SH.perturb(σ, p)
        σ_rec = SH.monte_carlo(σ_new, J; nsweeps = nsweeps, earlystop = earlystop, β = β, annealing = annealing)
        mf[i] = SH.overlap(σ_rec, σ)
    end
    show && show_hist(mf, N, α, nbins)
    save && save_data(mf, N, α, p, savedir)
end
return mf
end

function generate_intermediate_patterns(σ1, σ2)#, J)
    diff = findall(x -> x == -1, σ1 .* σ2)
    #energies = zeros(length(diff) + 2)
    #energies[1] = SH.energy(J, σ1)
    #energies[end] = SH.energy(J, σ2)
    new_patterns = zeros(N, length(diff))
    σ_new = copy(σ1)
    for i in 1:length(diff)
        σ_new[diff[i]] *= -1
        new_patterns[:, i] = σ_new
        #println(SH.overlap(σ_new, σ1))
        #energies[i+1] = SH.energy(J, σ_new)
    end
    return new_patterns
end

