include("../continuous_hopfield.jl")
using Statistics, LinearAlgebra, Plots, DelimitedFiles

function one_reconstruction_probability(N::Int, pp::AbstractVector, α, nsamples;
    nsweeps = 100,
    β = 10,
    thr = 0.1,
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
            if d <= thr
                success[sample] = 1
                #count += 1
            end

        end
        probs[i] = Statistics.mean(success)
        error_bars[i] = Statistics.std(success)/sqrt(nsamples)
    end

    return probs, error_bars
end

function reconstruction_probability(NN::AbstractVector,
    α;
    pp::AbstractVector = range( 0.1, 0.9, length = 22 ),
    β = 10^3,
    nsamples = 5*10^2,
    thr = 0.1,
    λ=1,
    show = false,
    save = true)

    for N in NN 
        prob, error = one_reconstruction_probability(N, pp, α, nsamples;  λ = λ, thr = thr) 

        if show
            fig = plot(pp, prob, size = (500,300), markershape =:circle, label = "N = $N, α = $α",
            xlabel = "p", ylabel = "P_reconst") 
            display(fig)        
        end

        if save
            folder = replace(string(α),"." => "" )
            path = "julia_data/alpha_"*folder

            if isdir(path)
                io = open(path*"/probsN"*"$N"*".txt", "w") do io
                    writedlm(io, [pp prob error mag])
                end
            else
                mkdir(path)
                io = open(path*"/probsN"*"$N"*".txt", "w") do io
                    writedlm(io, [pp prob error mag])
                end
            end
        end
    end
    return       
end

