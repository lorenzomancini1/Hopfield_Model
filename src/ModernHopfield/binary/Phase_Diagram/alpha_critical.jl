include("../modern_hopfield_binary.jl")
using Statistics, LinearAlgebra, Plots
using DelimitedFiles, Random

function stationary_frequency(N::Int64, α::AbstractVector, nsamples::Int64;
    nsweeps = 100, β = 10, earlystop = 0, λ = 1, m0 = 0.9)

    # this function computes the probability for a pattern to remain within m = 0.9 after a monte carlo run
    # for different values of α

    nα = length(α) # take a suitable range of α

    freqs, ferrors, mags, merrors = zeros(nα), zeros(nα), zeros(nα), zeros(nα) # initialize vectors for frequencies and final mags (with corresponding errors)

    for i in 1:nα # loop over α range
        mi, mf = zeros(nsamples), zeros(nsamples) # initialize vectors for initial and final mags
        M  = round(Int, exp(N * α[i])) # compute the number of patterns to store
        for sample in 1:nsamples # loop over nsamples
            ξ = MHB.generate_patterns(M, N) # generate patterns      
            k = rand(1:M) 
            σ = ξ[:, k] # take a random pattern
            mi[sample] = MHB.overlap(σ, σ) # compute initial overlap (this is necessary only if we start from a perturbed pattern)

            σ_rec = MHB.monte_carlo(σ, ξ;
             nsweeps = nsweeps, earlystop = earlystop, β = β, λ = λ) # run the monte carlo
            mf[sample] = MHB.overlap(σ_rec, σ) #compute final mag
        end
        success    = mf .>= m0 # if the final mag is greater than m0 = 0.9 it means that the patterns did not escape after a monte carlo run
        freqs[i]   = mean(success) # compute the probability to "remain there"
        ferrors[i] = std(success) / sqrt(nsamples)
        
        mags[i]   = mean(mf)
        merrors[i]= std(mf) / nsamples
    end
    return freqs, ferrors, mags, merrors            
end

function plotf(N::Int, λ::Float64, α::AbstractVector, f::AbstractVector)
    # function to plot the fit of probability
    fig = plot(α, f, size = (500,300), markershape =:circle, label = "N = $N, λ = $λ",
                    xlabel = "α", ylabel = "prob", legend =:bottomright) 
    display(fig)
    nothing
end

function savedata(N::Int, λ::Float64, data::AbstractMatrix; dir = "julia_data")
    # function to save data
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

function simulate(NN::AbstractVector, dd::AbstractVector;
    m0 = 0.9, # one_rec_freq params
    nsweeps = 100, β = 10^2, earlystop = 0, # monte carlo params
    save = true, show = false, savedir = "julia_data",
    verbose = true)

    # function to simulate for different values of N and λ
    # dd is a vector of tuples like [(λ, α)] where the first element is λ and the second one is a range of α

    # similarly, NN is a vector of tuples like [(N, nsamples)]
    for d in dd
        λ, α = d[1], d[2]
        verbose && println("------------------λ = $λ------------------")
        for n in NN
            N, nsamples = n[1], n[2]
            f, ferr, m, merr = stationary_frequency(N, α, nsamples;
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
