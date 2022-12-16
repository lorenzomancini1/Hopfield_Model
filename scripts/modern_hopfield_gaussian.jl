using LinearAlgebra, Random, Statistics
using DataFrames, CSV
using BenchmarkTools
using OrderedCollections: OrderedDict
using DrWatson
using OnlineStats
using ThreadsX
using Hopfield_Model: MHC
using Hopfield_Model: MyStats, mean_with_err, check_filename

BLAS.set_num_threads(1)
# BLAS.set_num_threads(Sys.CPU_THREADS)

function single_run_gd(; N = 50,
                    α = 0.2,
                    η = 0.5,
                    λ = 0.1,
                    nsamples = 100, 
                    maxsteps = 1000)

    M = round(Int, exp(α * N))
    stats = mapreduce(MyStats(), 1:nsamples) do _ 
        ξ = MHC.generate_patterns(N, M)
        σ0 = ξ[:,1]
        σ, res = MHC.gradient_descent(σ0, ξ; λ, η, maxsteps);
        return last(res)            
    end
    return mean_with_err(stats)
end

function span_gd(;
        N = 40,
        α = 0.2,
        η = 0.5,
        λ = [0.1:0.1:1.0;],
        maxsteps = 1000,
        nsamples = 100, 
        resfile = "res_gaussian.csv",
        )
    
    params_list = dict_list(OrderedDict(:N => N, :α=> α, :λ => λ, :nsamples => nsamples))
    allres = Vector{Any}(undef, length(params_list))
    
    ThreadsX.foreach(enumerate(params_list)) do (i, p)
        stats = single_run_gd(; p..., η, maxsteps)
        allres[i] = NamedTuple(merge(p, stats))
    end

    allres = DataFrame(allres)
    if resfile != ""
        respath = datadir("exp_gradient_descent_gaussian", resfile)
        respath = check_filename(respath)
        CSV.write(respath, allres)
    end
    return allres
end

@time span_gd(resfile="run.csv", N=40, nsamples=100, α=0.2)
