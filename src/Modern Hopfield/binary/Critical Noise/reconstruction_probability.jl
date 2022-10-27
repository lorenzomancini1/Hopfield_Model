include("../modern_hopfield_binary.jl")
using Statistics, LinearAlgebra, Plots, DelimitedFiles

function one_reconstruction_probability(N::Int, pp::AbstractVector, α, nsamples;
    nsweeps = 100,
    β = 10,
    earlystop = 0,
    thr = 0.95,
    λ = 1)

    M = round(Int, exp(N * α)) #compute M

    len_pp = length(pp) # number of p to compute
    probs = zeros(len_pp) # initialize array for reconstruction frequencies
    error_bars = zeros(len_pp) # initialize array for errors
    magnetization = zeros(len_pp) # initialize array for magnetization

    for i in 1:len_pp # loop over perturb probabilities
        success = zeros(nsamples) # array to contain the number of successes      
        #count = 0
        ms = zeros(nsamples)
        
        for sample in 1:nsamples
            ξ = MHB.generate_patterns(M, N)
            #J = MHB.store(ξ)
            
            k = rand(1:M)
            σ = ξ[:, k]
            σ_pert = MHB.perturb(σ, pp[i])
                        
            σ_rec = MHB.monte_carlo(σ_pert, ξ;
                                    nsweeps = nsweeps, earlystop = earlystop,
                                    β = β, λ = λ)
            
            m = MHB.overlap(σ_rec, σ) / N
            #print(m)
            if m >= thr
                success[sample] = 1
                #count += 1
            end
            ms[sample] = m
        end
        probs[i] = Statistics.mean(success)
        error_bars[i] = Statistics.std(success)/sqrt(nsamples)
        magnetization[i] = Statistics.mean(ms)
    end

    return probs, error_bars, magnetization
end

function reconstruction_probability(NN::AbstractVector,
    α;
    pp::AbstractVector = range( 0.14, 0.56, length = 22 ),
    nsweeps = 100,
    β = 10^3,
    nsamples = 5*10^2,
    earlystop = 0,
    thr = 0.95,
    λ=1,
    show = false,
    save = true)

    for N in NN 
        prob, error, mag = one_reconstruction_probability(N, pp, α, nsamples; 
        nsweeps = nsweeps, β = β, earlystop = earlystop, thr = thr, λ = λ) 

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