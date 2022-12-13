@testset "logsumexp" begin
    x = randn(100)
    @test MHC.logsumexp(x) ≈ log(sum(exp.(x)))
end

@testset "grad_energy" begin
    N = 10
    M = 5
    λ = 2
    ξ = MHC.generate_patterns(N, M)
    σ0 = randn(N)
    fdm = FiniteDifferences.central_fdm(5, 1)
    g = FiniteDifferences.grad(fdm, σ -> MHC.energy(σ, ξ, λ), σ0)[1]
    @test g ≈ MHC.grad_energy(σ0, ξ, λ)
end

