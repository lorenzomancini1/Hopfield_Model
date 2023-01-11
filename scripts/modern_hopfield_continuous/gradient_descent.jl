using LinearAlgebra, Random, Statistics
using DataFrames, CSV
using BenchmarkTools
using DrWatson
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
                    maxsteps = 1000, 
                    ξtype = :gaussian)

    M = round(Int, exp(α * N))
    stats = mapreduce(Stats(), 1:nsamples) do _
        ξ = MHG.generate_patterns(N, M; ξtype)
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
        ξtype = :gaussian,
        resfile = savename("run_"*gethostname(), (; N, α, nsamples), "csv", digits=4),
        respath = datadir("raw", "modern_hopfield_xgaussian_$(ξtype)", splitext(basename(@__FILE__))[1]), 
        )

    if resfile != "" && resfile !== nothing
        makepath(respath)
        resfile = joinpath(respath, resfile)
        resfile = check_filename(resfile) # appends a number if the file already exists
        touch(resfile)
    end

    params_list = cartesian_list(; N, α, λ, nsamples)
    
    lck = ReentrantLock()
    df = DataFrame()

    ThreadsX.foreach(params_list) do p
        stats = singlerun_gd(; p..., η, maxsteps)

        lock(lck) do 
            push!(df, merge(p, stats))
            if resfile != "" && resfile !== nothing
                CSV.write(resfile, df)
            end
        end
    end
    
    return df
end

# @time span_gd(resfile="run.csv", N=40, nsamples=100, α=0.2)
