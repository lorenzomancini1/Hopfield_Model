using Hopfield_Model: MHC
using LinearAlgebra, Random, Statistics
using DataFrames, CSV
using Plots, StatsPlots
using BenchmarkTools
using DataStructures
using DrWatson
# using ProfileView: @profview



function single_run(
        N = 10,
        α = 0.2,
        η = 0.5,
        maxsteps = 1000,
        nsamples = 10, 
        resfile = datadir("gradient_descent", "res_gaussian.csv")
        )
    
    M = round(Int, exp(α * N))
    allres = DataFrame()
    for λ in 0.1:0.1:1.0
        avg = OrderedDict{Symbol,Any}(:λ => λ, :nsamples => nsamples)
        for _ in 1:nsamples
            ξ = MHC.generate_patterns(N, M)
            σ0 = ξ[:,1]
            σ, res = MHC.gradient_descent(σ0, ξ; λ, η, maxsteps);
            update_average!(avg, last(res))
        end
        avg = finalize_average(avg)
        push!(allres, avg)
    end

    if resfile != ""
        mkpath(dirname(resfile))
        if isfile(resfile)
            df = CSV.read(resfile)
            append!(df, DataFrame(allres))
        else
            df = DataFrame(allres)
        end
        CSV.write(resfile, df)
    end
    return allres
end

function update_average!(avg::AbstractDict, nt)
    for (k, v) in pairs(nt)
        if !haskey(avg, k)
            avg[k] = Series(Mean(), Variance())
        end
        fit!(avg[k], v)
    end
end

function finalize_average(avg)
    d = OrderedDict{Symbol,Any}()
    for (k, v) in avg
        if v isa OnlineStat
            d[k] = value(v.stats[1])
            d[Symbol(k, "_err")] = sqrt(value(v.stats[2]) / v.stats[2].n)
        else
            d[k] = v
        end
    end
    return d
end

    single_run()