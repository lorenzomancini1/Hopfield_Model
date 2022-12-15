using Test, OnlineStats
using Hopfield_Model:MyStats, mean_with_err
using OrderedCollections: OrderedDict

@testset "mystats.jl" begin
    s = MyStats()
    OnlineStats.fit!(s, (a = 1, b = 2))
    OnlineStats.fit!(s, (a = 2, b = 3))
    @test mean(s) == OrderedDict(:a => 1.5, :b => 2.5)
    @test var(s) == OrderedDict(:a => 0.5, :b => 0.5)
    @test mean_with_err(s) == OrderedDict(:a => 1.5, :a_err => 0.5, :b => 2.5, :b_err => 0.5)

    @test mean(s) == OnlineStats.value(s)
    OnlineStats.fit!(s, [(a = 2, b = 3), (a = 2, b = 3)])
    @test OnlineStats.nobs(s) == 4
    @test length(s) == 2

    s1 = deepcopy(s)
    s2 = deepcopy(s)
    merge!(s1, s2)
    @test OnlineStats.nobs(s1) == 2 * OnlineStats.nobs(s)
end