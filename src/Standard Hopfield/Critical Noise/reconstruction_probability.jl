include("../standard_hopfield.jl")
using Statistics, LinearAlgebra, Plots
using DelimitedFiles, Random

function one_reconstruction_frequency(N::Int64, α::Float64, nsamples::Int64, pp::AbstractVector;
    nsweeps = 100, β = 10, earlystop = 0, m0 = 0.8)

    M  = round(Int, N * α)
    np = length(pp)

    freqs, ferrors, mags, merrors = zeros(np), zeros(np), zeros(np), zeros(np)

    for i in 1:np
        overlaps = zeros(nsamples)
        
        for sample in 1:nsamples
            ξ = SH.generate_patterns(M, N)
            J = SH.store(ξ)
            
            k = rand(1:M)
            σ = ξ[:, k]
            σ_pert = SH.perturb(σ, pp[i])
            
            σ_rec = SH.monte_carlo(σ_pert, J; nsweeps = nsweeps, earlystop = earlystop, β = β)
            overlaps[sample] = SH.overlap(σ_rec, σ)
        end
        success   = map(x -> x >= m0, overlaps)
        freqs[i]  = mean(success)
        ferrors[i] = std(success) / sqrt(nsamples)
        
        mags[i]   = mean(overlaps)
        merrors[i]= std(overlaps) / nsamples
    end
    return freqs, ferrors, mags, merrors            
end

function plotf(N::Int, α::Float64, pp::AbstractVector, f::AbstractVector)
    fig = plot(pp, f, size = (500,300), markershape =:circle, label = "N = $N, α = $α",
                    xlabel = "p", ylabel = "probs") 
    display(fig)
    return nothing
end

function savedata(N::Int, α::Float64, data::AbstractMatrix; dir = "julia_data")

    folder = replace(string(α),"." => "" )
    path = dir*"/alpha_"*folder

    if isdir(path)
        io = open(path*"/N"*"$N"*".txt", "w") do io
            writedlm(io, data)
        end
    else
        mkpath(path)
        io = open(path*"/N"*"$N"*".txt", "w") do io
            writedlm(io, data)
        end
    end
    return nothing
end

function reconstruction_frequencies(dd::AbstractVector, αα::AbstractVector;
    pp::AbstractVector = range( 0.14, 0.56, length = 22 ), m0 = 0.8, # one_rec_freq params
    nsweeps = 100, β = 15, earlystop = 0, # monte carlo params
    save = true, show = false, savedir = "julia_data")

    for α in αα
        for d in dd
            N, nsamples = d[1], d[2]
            f, ferr, m, merr = one_reconstruction_frequency(N, α, nsamples, pp; nsweeps = nsweeps, β = β, earlystop = earlystop, m0 = m0)
            
            data = [f ferr m merr]
            show == true && plotf(N, α, pp, f)
            save == true && savedata(N, α, data; dir = savedir)
            
        end    
    end
    return nothing
end