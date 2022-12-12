include("../standard_hopfield.jl")
include("../../Modern Hopfield/continuous/continuous_hopfield.jl")
using Statistics, LinearAlgebra, Plots
using DelimitedFiles, Random
using ProfileView
using BenchmarkTools

#export matrix_factorization_experiment, recover_global_minimum, experiment_global_minimum

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

    #ξ_new = zeros(Int, N, M)
    J_new = copy(J)
    M_new = 0 
    for m in 1:M
        for _ in 1:ntrials
            σ = SH.init_pattern(N)
            σ_rec = SH.monte_carlo(J_new, σ; nsweeps = nsweeps, earlystop = 0, β = 100, annealing)
            overlaps = (σ_rec' * ξ) ./ N # qua Marc ci mette l'energia analitica (vedi Amit)
            if maximum(abs, overlaps) >= 0.9#5
                # println("success")
                J_new -= σ_rec * σ_rec' ./ N
                J_new[diagind(J_new)] .= 0
                M_new += 1
                #ξ_new[:,m] .= σ_rec
                break
            end
            # println("fail: $(overlaps)")
            # p = histogram(overlaps, bins=-1:0.1:1)
        end
    end
    return M_new / M #ξ_new, 
end

function recover_global_minimum(J;
    nsweeps = 100,
    ntrials = 10,
    nrestarts = 1,
    annealing = false,
    λ = 1,
    info = false)

    N = size(J, 1)
    J_new = copy(J)
    σfinal = rand([-1,1], N)
    Efinal = SH.energy(J, σfinal) / N
    
    for r in 1:nrestarts
        σ = rand([-1,1], N)
        E = SH.energy(J, σ) / N

        #function report(r, t)
        #    println("restart=1, trial=$t: E=$(E) E=$(Efinal)")
        #end

        # verbose > 0 && report(r, 0)
        for t in 1:ntrials
            σnew = SH.monte_carlo(J_new, σ; nsweeps, earlystop = 0, β = 10, annealing)
            σnew = SH.monte_carlo(J, σnew; nsweeps, earlystop = 0, β = 10, annealing)
            Enew = SH.energy(J, σnew) / N
            
            if info
                @info "restart=$(r) trial=$(t)" E Enew SH.overlap(σ, σnew) Efinal
            end
            if Enew > E
                break
            end
            σ = σnew
            E = Enew
            J_new -= λ * σnew * σnew' ./ N
            J_new[diagind(J_new)] .= 0             # verbose > 0 && report(t)
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
        seed = -1,
        info = false)

    seed > 0 && Random.seed!(seed)
    M = round(Int, N * α)
    ξ = SH.generate_patterns(M, N)
    J = SH.store(ξ)

    σ, E = recover_global_minimum(J; nsweeps, nrestarts, ntrials, annealing, λ, info)
    overlaps = (σ' * ξ) ./ N
    #@show overlaps maximum(abs, overlaps)
    success = maximum(abs, overlaps) >= 0.95
    return success
end

function one_factorization_probability(; α::Float64 = 0.04, NN::AbstractVector = [100, 200, 300, 400, 500], nsamples::Int = 200,
    nsweeps = 100, nrestarts = 1, ntrials = 1, 
    λ = 1., β = 10^4, annealing = false)

    len_N = length(NN)
    probs = zeros(len_N)
    error_bars = zeros(len_N)

    for i in 1:len_N
        n = NN[i]
        probs_over_samples = zeros(nsamples)
        
        for sample in 1:nsamples
            success = experiment_global_minimum(; N = n, α = α, nsweeps = nsweeps, ntrials = ntrials,
            nrestarts = nrestarts, annealing = annealing, λ = λ, seed = -1)
            if success
                probs_over_samples[sample] = 1
            end
        end
        probs[i] = Statistics.mean(probs_over_samples)
        error_bars[i] = Statistics.std(probs_over_samples)/sqrt(nsamples)
    end
    return probs, error_bars
end

function factorization_probability(αα::AbstractVector, NN::AbstractVector;
    nsamples = 500, nrestarts = 1, nsweeps = 100, ntrials = 1, λ = 1., β = 10^4,
    annealing = false, show = false, save = true, rootdir = "./")
    probs = zeros(length(NN))
    for α in αα
        probs, errors = one_factorization_probability(; α = α, NN = NN, nsamples = nsamples,
        nsweeps = nsweeps, nrestarts = nrestarts, ntrials = ntrials,
        β = β, λ = λ, annealing = annealing) 

        if show 
            fig = plot(NN, probs, size = (500,300), markershape =:circle, label = "α = $α",
                yerrors = errors, xlabel = "N", ylabel = "P_infer")
            display(fig)
        end
        
        if save
            folder = replace(string(α),"." => "" )
            path = rootdir*"julia_data/factorization_prob/alpha_"*folder

            if isdir(path)
                io = open(path*"/data.txt", "w") do io
                    writedlm(io, [NN probs errors])
                end
            else
                mkdir(path)
                io = open(path*"/data.txt", "w") do io
                    writedlm(io, [NN probs erros])
                end
            end
            
        end
    end
    return probs
end

function stopping_time(; α = 0.04, N = 100, maxsweeps = 10^3,
    save = false, rootdir="./")

    stop = 1
    
    sweeps_range = 10 .^ range( log10(10), log10(maxsweeps), length = 100 )
    sweeps_range = round.(Int, sweeps_range)

    for nsweeps in sweeps_range#range(start = 10, step = 10, stop = 10^3)
        p, _ = one_factorization_probability(; NN = [N], α = α, β = 10^8, nsamples = 500,
        nsweeps = nsweeps, annealing = true)
        stop = nsweeps
        if p[1] >= 0.98
            break
        end
    end

    if save
        folder = replace(string(α),"." => "" )
        path = rootdir*"julia_data/stopping_times/alpha_"*folder

        if isdir(path)
            io = open(path*"/stop"*"$N"*".txt", "w") do io
                writedlm(io, [stop])
            end
        else
            mkdir(path)
            io = open(path*"/stop"*"$N"*".txt", "w") do io
                writedlm(io, [stop])
            end
        end
        
    end
    return stop
end

function compute_norm(J1::AbstractMatrix, σ::AbstractVector, N::Int, M::Int; diagonal = true)
    ξ = reshape(σ, (N, M))
    if diagonal
      J2 = SH.store(ξ)
    else
      J2 = (ξ * ξ') ./ N
    end

    d = norm(J1 - J2)
    return d
end

function initialize!(J, M, N)
    idxs = findall(x->x != 0, J)
    ξ_new = SH.generate_patterns(M, N)
    length(idxs)
    for idx in 1:round(Int, length(idxs)/2)
        i = idxs[idx][1]
        j = idxs[idx][2]
        if J[i, j] < 0
           for p in 1:M 
                ξ_new[j ,p] = -ξ_new[i, p]
            end
        else
            for p in 1:M 
                ξ_new[j ,p] = ξ_new[i, p] 
            end
        end
    end

    return reshape(ξ_new, (N*M))
end

function one_matrix_sweep!(J, σ, N, M, d_in; β = 10^8, diagonal = true)
    n_neurons = N*M
    flip = 0
    for i in 1:n_neurons
        k = rand(1:n_neurons)
        σ[k] *= -1
        d = compute_norm(J, σ, N, M; diagonal = diagonal)
        Δd = d - d_in
        if Δd < 0 || rand() < exp(-β*Δd)
            d_in = d
            flip += 1
        else
            σ[k] *= -1 #reject the flip
        end
    end
    return σ, d_in, flip/n_neurons
end

function matrix_monte_carlo(J::AbstractMatrix, M::Int; nsweeps = 100, earlystop = 0, β = 10^8, diagonal = true, annealing = false)
    N = size(J, 1)
    # instead of considering a matrix ξ of size (N, M), we consider a vector of size
    # (N*M, 1)
    σ = rand([-1, 1], N*M)
    
    distances = zeros(nsweeps + 1)
    d_in = compute_norm(J, σ, N, M)
    # then we run the Monte Carlo on this "pattern"
    distances[1] = d_in

    if annealing
      Tf = 1 / β
      T = range(100, Tf, length = nsweeps)
      β_n = 1 ./ T

      for i in 1:nsweeps
        σ, d_in, fliprate = one_matrix_sweep!(J, σ, N, M, d_in; diagonal = diagonal, β = β_n[i]  )
          distances[i+1] = d_in
          fliprate <= earlystop && break
      end
    else
      for i in 1:nsweeps
          σ, d_in, fliprate = one_matrix_sweep!(J, σ, N, M, d_in; diagonal = diagonal)
          distances[i+1] = d_in
          fliprate <= earlystop && break
      end
    end

    ξ_rec = reshape(σ, (N, M))
    J_rec = SH.store(ξ_rec)
    success = J_rec == J
    return ξ_rec, distances, success
end

function potential(σi)
    return (σi - 1)^2 * (σi + 1)^2   
end

function loss(σ, J; λ = 1)
    return (σ' * J * σ)/2 + λ * sum(map(potential, σ))
end

function loss_gradient(σ, J; λ = 1)
   return (J * σ) + λ * map(x -> 2*(x-1)*(x+1)^2 + 2*(x-1)^2*(x+1), σ) 
end

function gradient_descent(σ, J; nsteps = 100, λ = 1, γ = 0.05)
    λλ = range(λ, 10^(0), length = nsteps)
    γγ = range(γ, 10^(-4), length = nsteps)
    println(loss(σ, J))
    
    for i in 1:nsteps
        σ -= γγ[i] * loss_gradient(σ, J; λ = λλ[i])
        #σ -= loss_gradient1(σ, J)
        println(loss(σ, J))
    end
    return σ
end
