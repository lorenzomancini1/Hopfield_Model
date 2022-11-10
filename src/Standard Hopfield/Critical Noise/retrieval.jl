include("../standard_hopfield.jl")
using Statistics, LinearAlgebra, Plots
using DelimitedFiles, Random

function one_retrieval_frequency(N::Int64, α::Float64, nsamples::Int64, pp::AbstractVector;
    nsweeps = 100, β = 10, earlystop = 0, annealing = -1, m0 = 0.8)

    M  = round(Int, N * α)
    np = length(pp)

    freqs, ferrors, mags, merrors = zeros(np), zeros(np), zeros(np), zeros(np)

    for i in 1:np
        mi, mf = zeros(nsamples), zeros(nsamples)
        
        for sample in 1:nsamples
            ξ = SH.generate_patterns(M, N)
            J = SH.store(ξ)
            
            k = rand(1:M)
            σ = ξ[:, k]
            σ_pert = SH.perturb(σ, pp[i])
            mi[sample] = SH.overlap(σ_pert, σ)

            σ_rec = SH.monte_carlo(σ_pert, J;
             nsweeps = nsweeps, earlystop = earlystop, β = β, annealing = annealing)
            mf[sample] = SH.overlap(σ_rec, σ)
        end
        success   = mf .>= m0
        freqs[i]  = mean(success)
        ferrors[i] = std(success) / sqrt(nsamples)
        
        mags[i]   = mean(mf)
        merrors[i]= std(mf) / nsamples
    end
    return freqs, ferrors, mags, merrors            
end

function plotf(N::Int, α::Float64, pp::AbstractVector, f::AbstractVector)
    fig = plot(pp, f, size = (500,300), markershape =:circle, label = "N = $N, α = $α",
                    xlabel = "p", ylabel = "probs") 
    display(fig)
    nothing
end

function savedata(N::Int, α::Float64, data::AbstractMatrix; dir = "julia_data")

    folder = replace(string(α),"." => "" )
    path = dir*"/alpha_"*folder

    if isdir(path)
        io = open(path*"/N"*"$N"*".txt", "w") do io
            write(io, "ciao, questa è una prova\n")
            writedlm(io, data)
        end
    else
        mkpath(path)
        io = open(path*"/N"*"$N"*".txt", "w") do io
            write(io, "ciao, questa è una prova\n")
            writedlm(io, data)
            #write(io, "ciao, questa è una prova")
        end
    end
    nothing
end

function simulate_retrieval(NN::AbstractVector, dd::AbstractVector;
    m0 = 0.8, # one_rec_freq params
    nsweeps = 100, β = 15, earlystop = 0, annealing = -1,# monte carlo params
    save = true, show = false, savedir = "julia_data",
    verbose = true)

    for d in dd
        α, pp = d[1], d[2]
        verbose && println("------------------α = $α------------------")
        for n in NN
            N, nsamples = n[1], n[2]
            f, ferr, m, merr = one_retrieval_frequency(N, α, nsamples, pp;
                                nsweeps = nsweeps, β = β, earlystop = earlystop, annealing = annealing,
                                 m0 = m0)
            verbose && println("N = $N ---> Done!")
            data = [pp f ferr m merr]
            show == true && plotf(N, α, pp, f)
            save == true && savedata(N, α, data; dir = savedir)
            
        end    
    end
    nothing
end