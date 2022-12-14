module MHC

using LinearAlgebra, Random, Statistics, Distributions, KahanSummation, LaTeXStrings, Plots
using DelimitedFiles
export init_pattern, overlap, generate_patterns, energy

init_pattern(N) = randn(N)

overlap(σ1::AbstractVector, σ2::AbstractVector) = σ1 ⋅ σ2 / length(σ1)
sqdistance(σ1::AbstractVector, σ2::AbstractVector) = norm(σ1 .- σ2)^2 / length(σ1)

generate_patterns(N, M) = randn(N, M)

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
    ws[imax] = 0 # in order to use log1p later
    return xmax + log1p(sum_kbn(ws))
end

function energy(σ::AbstractVector, ξ::AbstractMatrix, λ)
    N, M = size(ξ)
    x = (λ .* (σ' * ξ)) |> vec 
    lse = -logsumexp(x) / λ
    return lse + dot(σ,σ) / 2
end

function grad_energy(σ::AbstractVector, ξ::AbstractMatrix, λ)
    N, M = size(ξ)
    x = (λ .* (σ' * ξ)) |> vec
    return σ  - softmax_expect(x, ξ)
end

function softmax(x::AbstractVector)
    xmax = maximum(x)
    ws = exp.(x .- xmax)
    s = sum_kbn(ws)
    return ws ./ s
end

function softmax_expect(x::AbstractVector, o::AbstractVector)
   return sum_kbn(softmax(x) .* o)
end

function softmax_expect(logits::AbstractVector, o::AbstractMatrix)
    N, M = size(o)
    @assert M == length(logits)
    ws = softmax(logits)
    return mapslices(sum_kbn, ws .* o', dims = 1) |> vec
    
    # y = zeros(N)
    # wso = ws .* o'
    # @assert size(wso) == (M, N)
    # # wso = zeros(M)
    # for i=1:N
    #     # wso .= ws .* vec(view(o, i, :))
    #     y[i] = sum_kbn(wso[:,i])
    # end
    # return y
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

function gradient_descent(σ0::AbstractVector, ξ::AbstractArray;
        λ = 1, 
        η=0.01, 
        xtol = 1e-6, 
        maxsteps=1000)
    σ = deepcopy(σ0)
    N = length(σ)

    res = []
    function saveres!(t)
        push!(res, (; t, E = energy(σ, ξ, λ) / N, Δ0 = sqdistance(σ, σ0)))
    end

    saveres!(0)
    for t in 1:maxsteps
        σ_new = σ .- η .* grad_energy(σ, ξ, λ)
        Δ = sqdistance(σ_new, σ)
        σ .= σ_new
        saveres!(t)
        √Δ < xtol && break
    end
    return σ, res
end

end #module
