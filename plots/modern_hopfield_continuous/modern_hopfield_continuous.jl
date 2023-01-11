using Hopfield_Model: MHG
using DrWatson
using DataFrames, CSV
using Plots, StatsPlots
using ScienceProjectTemplate: combine_results, subseteq


function read_data()
    # function data_analysis()
    respath = datadir("raw", "modern_hopfield_gaussian", "gradient_descent")
    files = [joinpath(respath, file) for file in readdir(respath)]
    dfs = [CSV.read(file, DataFrame) for file in files if endswith(file, ".csv")]
    df = vcat(dfs...)
    df = combine_results(df, by=1:3, cols=5:2:ncol(df), errs=6:2:ncol(df), col_n=:nsamples)
    return sort!(df, [:N, :α, :λ])
end

function make_plot(df; 
            α=0.2, 
            Ns = [50, 60, 70], 
            xlims=(0.15, 4),
            critline = true)

    dfs = Dict()
    for N in Ns
        dfs[N] = subseteq(df; α, N)
    end

    p = plot(; title = "Final dist. from init. cond. of GD. α=$α",
            xlabel = "λ", 
            ylabel = "Δ", 
            legend = :topright,
            xlims)
    
    # @df df_N40 plot!(:λ, :Δ0, yerr = :Δ0_err, label = "N = 40", msc=:auto)
    for N in Ns
        @df dfs[N] plot!(:λ, :Δ0, yerr = :Δ0_err, label = "N = $N", msc=:auto)
    end
    if critline
        vline!([MHG.λcrit(α)], label = "λcrit", ls=:dash, lw=1, color=:black)
    end
    return p
end


df = read_data()


make_plot(df, α=0.1, Ns=[50, 70, 100, 120], xlims=(0.05,0.3))
savefig(joinpath(@__DIR__, "fig_gd_α=0.1.pdf"))

make_plot(df, α=0.2, Ns=[50,60,70], xlims=(0.15,0.4))
savefig(joinpath(@__DIR__, "fig_gd_α=0.2.pdf"))

make_plot(df, α=0.4, Ns=[10, 15, 20, 25, 35], xlims=(0.,1.), critline=false)
savefig(joinpath(@__DIR__, "fig_gd_α=0.4.pdf"))
