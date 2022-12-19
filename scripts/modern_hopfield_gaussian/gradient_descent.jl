using LinearAlgebra, Random, Statistics
using DataFrames, CSV
using BenchmarkTools
using OrderedCollections: OrderedDict
using DrWatson
using OnlineStats
using ThreadsX
using Hopfield_Model: MHG
using ScienceProjectTemplate: Stats, mean_with_err, check_filename, cartesian_list

BLAS.set_num_threads(1)
# BLAS.set_num_threads(Sys.CPU_THREADS)

function singlerun_gd(; N = 50,
                    α = 0.2,
                    η = 0.5,
                    λ = 0.1,
                    nsamples = 100, 
                    maxsteps = 1000)

    M = round(Int, exp(α * N))
    stats = mapreduce(Stats(), 1:nsamples) do _
        ξ = MHG.generate_patterns(N, M)
        σ0 = ξ[:,1]
        σ, res = MHG.gradient_descent(σ0, ξ; λ, η, maxsteps)
        return last(res)            
    end
    return (; nsamples, mean_with_err(stats)...)
end

function span_gd(;
        N = 40,
        α = 0.2,
        η = 0.5,
        λ = [0.1:0.1:1.0;],
        maxsteps = 1000,
        nsamples = 100, 
        resfile = savename((; N, α, nsamples), "csv", digits=4),
        respath = datadir("raw", "modern_hopfield_gaussian", splitext(basename(@__FILE__))[1]), 
        )
    
    params_list = cartesian_list(; N, α, λ, nsamples)
    allres = Vector{Any}(undef, length(params_list))
    
    ThreadsX.foreach(enumerate(params_list)) do (i, p)
        stats = singlerun_gd(; p..., η, maxsteps)
        allres[i] = merge(p, stats)
    end

    allres = DataFrame(allres)
    if resfile != "" && resfile !== nothing
        path = joinpath(respath, resfile)
        path = check_filename(path) # appends a number if the file already exists
        CSV.write(path, allres)
    end
    return allres
end

# @time span_gd(resfile="run.csv", N=40, nsamples=100, α=0.2)
