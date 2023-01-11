include("../continuous_hopfield.jl")
using Statistics, LinearAlgebra, Plots, DelimitedFiles

function one_reconstruction_probability(N::Int, pp::AbstractVector, α, nsamples;
    d0 = 0.05,
    λ = 1)

    M = round(Int, exp(N * α)) #compute M

    len_pp = length(pp) # number of p to compute
    probs = zeros(len_pp) # initialize array for reconstruction frequencies
    error_bars = zeros(len_pp) # initialize array for errors

    for i in 1:len_pp # loop over perturb probabilities
        success = zeros(nsamples) # array to contain the number of successes      
        #count = 0
        
        for sample in 1:nsamples
            ξ = CH.generate_patterns(M, N)
            #J = MHB.store(ξ)
            
            k = rand(1:M)
            σ = ξ[:, k]
            σ_pert = CH.perturb(σ, pp[i])
                        
            σ_rec = CH.update(σ_pert, ξ; λ = λ)
            
            d = CH.distance(σ_rec, σ)
            #print(m)
            if d <= d0
                success[sample] = 1
                #count += 1
            end

        end
        probs[i] = Statistics.mean(success)
        error_bars[i] = Statistics.std(success)/sqrt(nsamples)
    end

    return probs, error_bars
end

function savedata(N::Int, λ::Float64, α::Float64, data::AbstractMatrix,
    nsamples::Int64, d0::Float64; dir = "julia_data")

    #folder = replace(string(λ),"." => "" )
    #subfolder = replace(string(α),"." => "" )
    #path = dir*"/lambda_"*folder*"/alpha_"*subfolder

    folder_lambda = replace(string(λ),"." => "" )
    folder_d0 = replace(string(d0),"." => "" )
    folder_alpha = replace(string(α),"." => "" )
    path = dir*"/q_"*folder_d0*"/lambda_"*folder_lambda*"/alpha_"*folder_alpha

    if isdir(path)
        io = open(path*"/N"*"$N"*".txt", "w") do io
            write(io, "nsamples = "*"$nsamples"*"d0 = "*"$d0"*"\n")
            writedlm(io, data)
        end
    else
        mkpath(path)
        io = open(path*"/N"*"$N"*".txt", "w") do io
            write(io, "nsamples = "*"$nsamples"*"d0 = "*"$d0"*"\n")
            writedlm(io, data)
        end
    end
    nothing
end

function plotf(N::Int, α::Float64, pp::AbstractVector, f::AbstractVector)
    fig = plot(pp, f, size = (500,300), markershape =:circle, label = "N = $N, α = $α",
                    xlabel = "p", ylabel = "probs") 
    display(fig)
    nothing
end

function simulate_retrieval(NN::AbstractVector, dd::AbstractVector;
    d0 = 0.05, # one_rec_freq params
    λ = 1.,# monte carlo params
    save = true, show = false, savedir = "julia_data",
    verbose = true)

    for d in dd
        α, pp = d[1], d[2]
        verbose && println("------------------α = $α------------------")
        for n in NN
            N, nsamples = n[1], n[2]
            f, ferr  = one_reconstruction_probability(N, pp, α, nsamples;  λ = λ, d0 = d0)
            verbose && println("N = $N ---> Done!")
            data = [pp f ferr]
            show == true && plotf(N, α, pp, f)
            save == true && savedata(N, λ, α, data, nsamples, d0; dir = savedir)
            
        end    
    end
    nothing
end