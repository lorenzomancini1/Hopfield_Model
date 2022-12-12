include("../standard_hopfield.jl")

using Statistics, LinearAlgebra, Plots
using DelimitedFiles, Random

function stationary_frequency(N::Int64, α::AbstractVector, nsamples::Int64;
    nsweeps = 100, β = 10, earlystop = 0, m0 = 0.9)

    # this function computes the probability for a pattern to remain within m = 0.9 after a monte carlo run
    # for different values of α

    nα = length(α) # take a suitable range of α

    freqs, ferrors, mags, merrors = zeros(nα), zeros(nα), zeros(nα), zeros(nα) # initialize vectors for frequencies and final mags (with corresponding errors)

    for i in 1:nα # loop over α range
        mi, mf = zeros(nsamples), zeros(nsamples) # initialize vectors for initial and final mags
        M  = round(Int, N*α[i]) # compute the number of patterns to store
        for sample in 1:nsamples # loop over nsamples
            ξ = SH.generate_patterns(M, N) # generate patterns      
            J = SH.store(ξ)
            k = rand(1:M) 
            σ = ξ[:, k] # take a random pattern
            #mi[sample] = SH.overlap(σ, σ) # compute initial overlap (this is necessary only if we start from a perturbed pattern)

            σ_rec = SH.monte_carlo(σ, J;
             nsweeps = nsweeps, earlystop = earlystop, β = β) # run the monte carlo
            mf[sample] = SH.overlap(σ_rec, σ) #compute final mag
        end
        success    = mf .>= m0 # if the final mag is greater than m0 = 0.9 it means that the patterns did not escape after a monte carlo run
        freqs[i]   = mean(success) # compute the probability to "remain there"
        ferrors[i] = std(success) / sqrt(nsamples)
        
        mags[i]   = mean(mf)
        merrors[i]= std(mf) / nsamples
    end
    return freqs, ferrors, mags, merrors            
end

function plotf(N::Int, α::AbstractVector, f::AbstractVector)
    # function to plot the fit of probability
    fig = plot(α, f, size = (500,300), markershape =:circle, label = "N = $N",
                    xlabel = "α", ylabel = "prob", legend =:bottomright) 
    display(fig)
    nothing
end

function savedata(N::Int, data::AbstractMatrix, nsamples, β; dir = "julia_data")
    # function to save data
    path = dir*"/alpha_"

    if isdir(path)
        io = open(path*"/N"*"$N"*".txt", "w") do io
            write(io, "nsamples = "*"$nsamples"*" β = "*"$β"*"\n")
            writedlm(io, data)
        end
    else
        mkpath(path)
        io = open(path*"/N"*"$N"*".txt", "w") do io
            write(io, "nsamples = "*"$nsamples"*" β = "*"$β"*"\n")
            writedlm(io, data)
        end
    end
    nothing
end

function simulate(NN::AbstractVector, αα::AbstractVector;
    m0 = 0.9, # one_rec_freq params
    nsweeps = 100, β = 10^2, earlystop = 0, # monte carlo params
    save = true, show = false, savedir = "julia_data",
    verbose = true)

    # function to simulate for different values of N and λ
    # dd is a vector of tuples like [(λ, α)] where the first element is λ and the second one is a range of α

    # similarly, NN is a vector of tuples like [(N, nsamples)]

        #λ, α = d[1], d[2]
        #verbose && println("------------------λ = $λ------------------")
    for n in NN
        N, nsamples = n[1], n[2]
        f, ferr, m, merr = stationary_frequency(N, αα, nsamples;
                            nsweeps = nsweeps, β = β, earlystop = earlystop,
                                m0 = m0)
        verbose && println("N = $N ---> Done!")
        data = [αα f ferr m merr]
        show == true && plotf(N, αα, f)
        save == true && savedata(N, data, nsamples, β; dir = savedir)
        
    end    

    nothing
end