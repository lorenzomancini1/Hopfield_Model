using LinearAlgebra, Random, Statistics
using DataFrames, CSV
using BenchmarkTools
using OrderedCollections: OrderedDict
using DrWatson
using OnlineStats
using ThreadsX
using Hopfield_Model: MHC
using Hopfield_Model: MyStats, mean_with_err, check_filename

function gradient_descent(;
        N = 10,
        α = 0.2,
        η = 0.5,
        λ = [0.1:0.1:1.0;],
        maxsteps = 1000,
        nsamples = 10, 
        resfile = "res_gaussian.csv",
        )
    
    params_list = dict_list(OrderedDict(:N => N, :α=> α, :λ => λ, :nsamples => nsamples))
    allres = Vector{Any}(undef, length(params_list))
    foreach(enumerate(params_list)) do (i, p)
        M = round(Int, exp(p[:α] * p[:N]))
        stats = ThreadsX.mapreduce(MyStats(), 1:p[:nsamples]) do _ 
            ξ = MHC.generate_patterns(p[:N], M)
            σ0 = ξ[:,1]
            σ, res = MHC.gradient_descent(σ0, ξ; λ=p[:λ], η, maxsteps);
            return res            
        end
        stats = mean_with_err(stats)
        allres[i] = NamedTuple(merge(p, stats))
    end

    allres = DataFrame(allres)
    if resfile != ""
        respath = datadir("gradient_descent", resfile)
        respath = check_filename(respath)
        CSV.write(respath, allres)
    end
    return allres
end


@btime gradient_descent(resfile="", N=10, nsamples=100)

