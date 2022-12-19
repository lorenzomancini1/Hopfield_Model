module MHG

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
    # return mapslices(sum_kbn, ws .* o', dims = 1) |> vec # type unstable
    
    wso = ws .* o'
    return [sum_kbn(wso[:,i]) for i=1:N]
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
