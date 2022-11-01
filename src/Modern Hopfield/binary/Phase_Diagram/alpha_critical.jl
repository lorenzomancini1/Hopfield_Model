include("../modern_hopfield_binary.jl")
using Statistics, LinearAlgebra, Plots
using DelimitedFiles, Random

function escape_frequency(N::Int64, α::AbstractVector, nsamples::Int64;
    nsweeps = 100, β = 10, earlystop = 0, λ = 1, m0 = 0.8)

    nα = length(α)

    freqs, ferrors, mags, merrors = zeros(nα), zeros(nα), zeros(nα), zeros(nα)

    Threads.@threads for i in 1:nα
        mi, mf = zeros(nsamples), zeros(nsamples)
        M  = round(Int, exp(N * α[i]))
        for sample in 1:nsamples
            ξ = MHB.generate_patterns(M, N)            
            k = rand(1:M)
            σ = ξ[:, k]
            mi[sample] = MHB.overlap(σ, σ)

            σ_rec = MHB.monte_carlo(σ, ξ;
             nsweeps = nsweeps, earlystop = earlystop, β = β, λ = λ)
            mf[sample] = MHB.overlap(σ_rec, σ)
        end
        success    = mf .< m0
        freqs[i]   = mean(success)
        ferrors[i] = std(success) / sqrt(nsamples)
        
        mags[i]   = mean(mf)
        merrors[i]= std(mf) / nsamples
    end
    return freqs, ferrors, mags, merrors            
end

function plotf(N::Int, λ::Float64, α::AbstractVector, f::AbstractVector)
    fig = plot(α, f, size = (500,300), markershape =:circle, label = "N = $N, λ = $λ",
                    xlabel = "α", ylabel = "prob", legend =:bottomright) 
    display(fig)
    nothing
end

function savedata(N::Int, λ::Float64, data::AbstractMatrix; dir = "julia_data")

    folder = replace(string(λ),"." => "" )
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
    nothing
end

function simulate_escape(NN::AbstractVector, dd::AbstractVector;
    m0 = 0.9, # one_rec_freq params
    nsweeps = 100, β = 10^2, earlystop = 0, # monte carlo params
    save = true, show = false, savedir = "julia_data",
    verbose = true)

    for d in dd
        λ, α = d[1], d[2]
        verbose && println("------------------λ = $λ------------------")
        for n in NN
            N, nsamples = n[1], n[2]
            f, ferr, m, merr = escape_frequency(N, α, nsamples;
                                nsweeps = nsweeps, β = β, earlystop = earlystop, λ = λ,
                                 m0 = m0)
            verbose && println("N = $N ---> Done!")
            data = [α f ferr m merr]
            show == true && plotf(N, λ, α, f)
            save == true && savedata(N, λ, data; dir = savedir)
            
        end    
    end
    nothing
end
