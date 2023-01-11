

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