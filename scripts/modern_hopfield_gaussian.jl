using Hopfield_Model: MHC
using LinearAlgebra, Random, Statistics
using DataFrames, CSV
using Plots, StatsPlots
using BenchmarkTools
using OrderedCollections: OrderedDict
using DrWatson
using OnlineStats
using Hopfield_Model: MyStats, mean_with_err

# using ProfileView: @profview



function gradient_descent(;
        N = 10,
        α = 0.2,
        η = 0.5,
        maxsteps = 1000,
        nsamples = 10, 
        resfile = "res_gaussian.csv",
        )
    
    M = round(Int, exp(α * N))
    allres = DataFrame()
    for λ in 0.1:0.1:1.0
        stats = MyStats()
        for _ in 1:nsamples
            ξ = MHC.generate_patterns(N, M)
            σ0 = ξ[:,1]
            σ, res = MHC.gradient_descent(σ0, ξ; λ, η, maxsteps);
            OnlineStats.fit!(stats, last(res))
        end
        stats = mean_with_err(stats)
        push!(allres, NamedTuple(stats))
    end

    if resfile != ""
        respath = datadir("gradient_descent", resfile)
        mkpath(dirname(respath))
        respath = check_filename(respath)
        CSV.write(respath, allres)
    end
    return allres
end

"""
Check if a file with the name exists, if so, append a number to the name.
"""
function check_filename(filename)
    i = 0
    _filename = filename
    while isfile(_filename)
        i += 1
        _filename = filename * "." * string(i)
    end
    filename = _filename
    return filename
end

# function update_average!(avg::AbstractDict, nt)
#     for (k, v) in pairs(nt)
#         if !haskey(avg, k)
#             avg[k] = Series(Mean(), Variance())
#         end
#         OnlineStats.fit!(avg[k], v)
#     end
# end

# function finalize_average(avg)
#     d = OrderedDict{Symbol,Any}()
#     for (k, v) in avg
#         if v isa OnlineStat
#             d[k] = value(v.stats[1])
#             d[Symbol(k, "_err")] = sqrt(value(v.stats[2]) / v.stats[2].n)
#         else
#             d[k] = v
#         end
#     end
#     return d
# end

gradient_descent(resfile="test.csv")

params_list = dict_list(Dict(:λ => [0.1:0.1:1.0;], :nsamples => 10))