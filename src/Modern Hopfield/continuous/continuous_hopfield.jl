module CH

using LinearAlgebra, Random, Statistics, Distributions, KahanSummation, LaTeXStrings, Plots
using DelimitedFiles

export init_pattern, overlap, generate_patterns, energy

function init_pattern(N)
    #σ = rand(1:5, N) #cambiare
    σ = randn(N)
    return σ
end

function overlap(σ1::AbstractVector, σ2::AbstractVector)
    return σ1 ⋅ σ2

end

function distance(σ1::AbstractVector, σ2::AbstractVector)
    return norm(σ1 - σ2)
end

function generate_patterns(M, N)
    #ξ = 1. + 4 * rand(N, M)
    ξ = randn(N, M)
    return ξ
end

function perturb(σ::AbstractVector, δ::Float64)
    N = length(σ)
    σ_new = copy(σ)
    noise = rand(Normal(0, δ), N)
    return σ_new + noise
end

function logsumexp(x::AbstractVector)
    imax = argmax(x)
    xmax = x[imax]
    ws = exp.(x .- xmax)
    ws[imax] = 0 # in order to se log1p later
    return xmax + log1p(sum_kbn(ws))
end

function energy(σ::AbstractVector, ξ::AbstractMatrix; β = 10, γ=1)
    M = size(ξ, 2)
    #N = length(σ)
    
    max_norm_sq = argmax(vec(sum(ξ.^2, dims = 1) ) )
    #print(max_norm_sq)
    se = 0
    
    for μ in 1:M
        se += exp(β * overlap(σ, ξ[:, μ]))
    end
    lse = log(se)/β
    energy = -lse + 0.5*γ*sum(σ.^2) + log(M)/β + 0.5*max_norm_sq
    
    if energy >= 10^4
        return 10^4
    else
        return energy
    end
end

function energy1(σ, ξ; λ = 1)
    M = size(ξ, 2)
    lse = logsumexp(ξ' * σ)/λ
    max_norm_sq = maximum(vec(sum(ξ.^2, dims = 1) ) )
    return -lse + 0.5 * max_norm_sq + log(M)/λ + 0.5 * (σ' * σ)
end

function softmax(x)
    imax = argmax(x)
    xmax = x[imax]
    ws = exp.(x .- xmax)
    s = sum_kbn(ws) - ws[imax]
    return ws ./ (1 + s)
end


function contourplot(; N = 40, α = 0.1, show = true, save = true, n1 = 150, n2 = 150)
    M = round(Int, exp(N*α))
    ξ = CH.generate_patterns(M, N)
    k1, k2, k3 = rand(1:M), rand(1:M), rand(1:M)
    σ1, σ2, σ3 = ξ[:, k1], ξ[:, k2], ξ[:, k3]
    
    ϵ1 = range(-1, 2, length = n1)
    ϵ2 = range(-1, 2, length = n2)
    Z = zeros( (length(ϵ1), length(ϵ2)) )
    
    function savecontour(data; path = "julia_data/contour")
        if isdir(path)
            io = open(path*"/contour.txt", "w") do io
                writedlm(io, data)
            end
        else
            mkpath(path)
            io = open(path*"/contour.txt", "w") do io
                writedlm(io, data)
            end
        end
    end

    for i in eachindex(ϵ1)
        a = ϵ1[i]
        for j in eachindex(ϵ2)
            σ_new = σ1 + a*(σ2 - σ1) + ϵ2[j]*(σ3 - σ1)
            Z[i, j] = CH.energy1(σ_new, ξ)
        end
    end

    data = [Z']
    fig = contour(ϵ1, ϵ2, Z', levels = 90, xlabel = L"\epsilon_1", ylabel = L"\epsilon_2")
    show && display(fig)
    save && savecontour(data)
end

function update(σ, ξ; λ = 1, nsweeps = 1)
    σ_rec = copy(σ)
    for _ in 1:nsweeps
        σ_rec = ξ * softmax(λ .* vec(ξ' * σ_rec))
    end
    return σ_rec
end

end
