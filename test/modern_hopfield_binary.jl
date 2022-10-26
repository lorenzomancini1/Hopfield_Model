

@testset "retrieval at low alpha and low lambda" begin
    N = 50
    α = 0.2
    M = round(Int, exp(α*N))
    @info (; N, α, M)
    # Random.seed!(17)
    ξ = MHB.generate_patterns(M, N)
    # σ0 = MHB.init_pattern(N)
    σ0 = ξ[:,1]
    @test size(ξ) == (N, M)
    @test length(σ0) == N

    σ =  MHB.monte_carlo(σ0, ξ; nsweeps = 10, earlystop = 0, β = 10^2, λ = 0.2)
    qs = vec(σ' * ξ) ./ N
    @test qs[1] == 1
end