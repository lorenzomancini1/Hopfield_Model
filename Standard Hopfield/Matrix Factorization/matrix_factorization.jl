include("standard_hopfield.jl")
using Statistics, LinearAlgebra, Plots
using DelimitedFiles, Random
using ProfileView
using BenchmarkTools

function matrix_factorization_experiment(;
        N = 1000,
        α = 0.04,
        nsweeps = 100,
        ntrials = 1,
        annealing = true, 
        seed=-1)
        
    seed > 0 && Random.seed!(seed)
    M = round(Int, N * α)
    ξ = SH.generate_patterns(M, N)
    J = SH.store(ξ)

    ξ_new = zeros(Int, N, M)
    J_new = copy(J)
    M_new = 0 
    for m in 1:M
        for _ in 1:ntrials
            σ = SH.init_pattern(N)
            σ_rec = SH.monte_carlo(J_new, σ; nsweeps = nsweeps, earlystop = 0, β = 100, annealing)
            overlaps = (σ_rec' * ξ) ./ N
            if maximum(abs, overlaps) >= 0.95
                # println("success")
                J_new -= σ_rec * σ_rec' ./ N
                J_new[diagind(J_new)] .= 0
                M_new += 1
                ξ_new[:,m] .= σ_rec
                break
            end
            # println("fail: $(overlaps)")
            # p = histogram(overlaps, bins=-1:0.1:1)
        end
    end
    return ξ_new, M_new / M
end

function recover_global_minimum(J;
    nsweeps = 100,
    ntrials = 10,
    nrestarts = 1,
    annealing = false,
    λ = 1, 
    verbose = 0)

    N = size(J, 1)
    J_new = copy(J)
    σfinal = rand([-1,1], N)
    Efinal = SH.energy(J, σfinal) / N
    


    for r in 1:nrestarts
        σ = rand([-1,1], N)
        E = SH.energy(J, σ) / N

        function report(r, t)
            println("restart=1, trial=$t: E=$(E) E=$(Efinal)")
        end

        # verbose > 0 && report(r, 0)
        for t in 1:ntrials
            σnew = SH.monte_carlo(J_new, σ; nsweeps, earlystop = 0, β = 10, annealing)
            σnew = SH.monte_carlo(J, σnew; nsweeps, earlystop = 0, β = 10, annealing)
            Enew = SH.energy(J, σnew) / N
            @info "restart=$(r) trial=$(t)" E Enew SH.overlap(σ, σnew) Efinal
            if Enew > E
                break
            end
            σ = σnew
            E = Enew
            J_new -= λ * σnew * σnew' ./ N
            # verbose > 0 && report(t)
        end
        if E < Efinal
            σfinal = σ
            Efinal = E
        end
    end
    return σfinal, Efinal
end


function experiment_global_minimum(;
        N = 1000,
        α = 0.04,
        nsweeps = 100,
        ntrials = 10,
        nrestarts = 1,
        annealing = false,
        λ = 1.,
        seed = -1)

    seed > 0 && Random.seed!(seed)
    M = round(Int, N * α)
    ξ = SH.generate_patterns(M, N)
    J = SH.store(ξ)

    σ, E = recover_global_minimum(J; nsweeps, nrestarts, ntrials, annealing, λ, verbose=1)
    overlaps = (σ' * ξ) ./ N
    @show overlaps maximum(abs, overlaps)
    success = maximum(abs, overlaps) >= 0.95
    return success
end

