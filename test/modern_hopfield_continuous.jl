@testset "logsumexp" begin
    x = randn(100)
    y = @inferred MHC.logsumexp(x)
    @test y ≈ log(sum(exp.(x)))
end

@testset "softmax_expect" begin
    N, M = 100, 1000
    logits = randn(M)
    ξ = randn(N, M)
    ws = @inferred MHC.softmax(logits)
    @test all(1 .≥ ws .≥ 0)
    @test sum(ws) ≈ 1
    ys = ws .* ξ'
    a = @inferred MHC.softmax_expect(logits, ξ)
    @test size(a) == (N,)
    @test a ≈ mapslices(sum, ys, dims=1) |> vec
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

