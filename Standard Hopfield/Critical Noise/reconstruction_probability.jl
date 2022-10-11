include("../standard_hopfield.jl")
using Statistics, LinearAlgebra, Plots
using DelimitedFiles, Random

function one_reconstruction_probability(N::Int, pp::AbstractVector, α, nsamples;
    nsweeps = 100,
    β = 10,
    earlystop = 0,
    thr = 0.95)

    M = round(Int, N * α)

    len_pp = length(pp)
    probs = zeros(len_pp)
    error_bars = zeros(len_pp)
    magnetization = zeros(len_pp)

    for i in 1:len_pp
        probs_over_samples = zeros(nsamples)        
        #count = 0
        ms = zeros(nsamples)
        
        for sample in 1:nsamples
            ξ = SH.generate_patterns(M, N)
            J = SH.store(ξ)
            
            k = rand(1:M)
            σ = ξ[:, k]
            σ_pert = SH.perturb(σ, pp[i])
                        
            σ_rec = SH.monte_carlo(J, σ_pert; nsweeps = nsweeps, earlystop = earlystop, β = β)
            
            m = SH.overlap(σ_rec, σ)
            #print(m)
            if m >= thr
                probs_over_samples[sample] = 1
                #count += 1
            end
            ms[sample] = m
        end
        probs[i] = Statistics.mean(probs_over_samples)
        error_bars[i] = Statistics.std(probs_over_samples)/sqrt(nsamples)
        magnetization[i] = Statistics.mean(ms)
    end

    return probs, error_bars, magnetization
end

function reconstruction_probability(NN::AbstractVector,
    α;
    pp::AbstractVector = range( 0.08, 0.58, length = 22 ),
    nsweeps = 100,
    β = 10^3,
    nsamples = 5*10^2,
    earlystop = 0,
    thr = 0.95,
    show = false,
    save = true)

    for N in NN 
        prob, error, mag = one_reconstruction_probability(N, pp, α, nsamples; 
        nsweeps = nsweeps, β = β, earlystop = earlystop, thr = thr) 

        if show
            fig = plot(pp, prob, size = (500,300), markershape =:circle, label = "N = $N, α = $α",
            yerrors = error, xlabel = "p", ylabel = "P_reconst") 
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