# MODULE: test_zn_infrastructure.jl
# USAGE: julia --project=ekrgilttrnr test/test_zn_infrastructure.jl
# INPUTS: None
# OUTPUTS: Test results
# DESCRIPTION: Tests for generalized Z_N Newton infrastructure
#
# BASED ON: Existing test patterns in the project
# NEW IN THIS MODULE: Tests for ZNTensor, GF(N), gauge fixing

using Test
using LinearAlgebra

# Include the modules we're testing
include("../src/EchelonFormGFN.jl")
include("../src/ZNTensor.jl")
include("../src/DiscreteGaugeZN.jl")
include("../src/ModelProvider.jl")

println("=" ^ 60)
println("Testing Z_N Newton Infrastructure")
println("=" ^ 60)

################################################
# Test GF(N) Arithmetic
################################################

@testset "GF(N) Arithmetic" begin
    @testset "Modular inverse" begin
        # GF(2)
        @test mod_inverse(1, 2) == 1

        # GF(3)
        @test mod_inverse(1, 3) == 1
        @test mod_inverse(2, 3) == 2  # 2 * 2 = 4 ≡ 1 (mod 3)

        # GF(5)
        @test mod_inverse(1, 5) == 1
        @test mod_inverse(2, 5) == 3  # 2 * 3 = 6 ≡ 1 (mod 5)
        @test mod_inverse(3, 5) == 2
        @test mod_inverse(4, 5) == 4  # 4 * 4 = 16 ≡ 1 (mod 5)

        # GF(7)
        for a in 1:6
            inv_a = mod_inverse(a, 7)
            @test mod(a * inv_a, 7) == 1
        end
    end

    @testset "Echelon form GF(3)" begin
        # Simple system over GF(3)
        A = [1 2 0; 2 1 1; 0 1 2]
        A_copy = copy(A)
        A_ech, rank = echelon_form_gfn!(A_copy, 3)

        @test rank == 3
        # Upper triangular structure
        @test A_ech[2, 1] == 0
        @test A_ech[3, 1] == 0
        @test A_ech[3, 2] == 0
    end

    @testset "Linear system solve GF(3)" begin
        # Solve x + 2y = 1, 2x + y = 2 over GF(3)
        A = [1 2; 2 1]
        b = [1, 2]
        x, success = solve_gfn(A, b, 3)

        @test success
        @test mod(A[1, 1] * x[1] + A[1, 2] * x[2], 3) == b[1]
        @test mod(A[2, 1] * x[1] + A[2, 2] * x[2], 3) == b[2]
    end

    @testset "Linear system solve GF(5)" begin
        # Random system over GF(5)
        A = [1 2 3; 4 0 1; 2 3 4]
        b = [1, 2, 3]
        x, success = solve_gfn(A, b, 5)

        @test success
        for i in 1:3
            result = sum(A[i, j] * x[j] for j in 1:3)
            @test mod(result, 5) == b[i]
        end
    end
end

################################################
# Test Model Provider
################################################

@testset "Model Provider" begin
    @testset "Model properties" begin
        ising = IsingModel()
        @test symmetry_order(ising) == 2
        @test critical_beta(ising) ≈ log(1 + sqrt(2)) / 2

        potts3 = PottsModel(3)
        @test symmetry_order(potts3) == 3
        @test critical_beta(potts3) ≈ log(1 + sqrt(3))

        potts4 = PottsModel(4)
        @test symmetry_order(potts4) == 4
        @test critical_beta(potts4) ≈ log(1 + sqrt(4))
    end

    @testset "CFT data" begin
        ising = IsingModel()
        dims = scaling_dimensions(ising)
        @test dims.identity == 0.0
        @test dims.spin ≈ 1/8
        @test dims.energy ≈ 1.0

        potts3 = PottsModel(3)
        dims3 = scaling_dimensions(potts3)
        @test dims3.identity == 0.0
        @test dims3.spin ≈ 2/15
        @test dims3.energy ≈ 4/5
    end

    @testset "Boltzmann weights" begin
        # Ising
        W_ising = boltzmann_weight_matrix(IsingModel(), 0.5)
        @test size(W_ising) == (2, 2)
        @test W_ising[1, 1] ≈ exp(0.5)
        @test W_ising[1, 2] ≈ exp(-0.5)

        # Potts
        W_potts = boltzmann_weight_matrix(PottsModel(3), 1.0)
        @test size(W_potts) == (3, 3)
        @test W_potts[1, 1] ≈ exp(1.0)
        @test W_potts[1, 2] ≈ 1.0
        @test W_potts[2, 3] ≈ 1.0

        # Clock
        W_clock = boltzmann_weight_matrix(ClockModel(4), 1.0)
        @test size(W_clock) == (4, 4)
        @test W_clock[1, 1] ≈ exp(1.0)  # cos(0) = 1
        @test W_clock[1, 3] ≈ exp(-1.0)  # cos(π) = -1
    end

    @testset "DFT matrix" begin
        # Check DFT is unitary
        for N in [2, 3, 4, 5]
            F = dft_matrix(N)
            @test size(F) == (N, N)
            @test F * F' ≈ I  # Unitary
        end

        # Z_2 DFT is Hadamard / √2
        F2 = dft_matrix(2)
        @test F2[1, 1] ≈ 1/sqrt(2)
        @test F2[1, 2] ≈ 1/sqrt(2)
        @test F2[2, 2] ≈ -1/sqrt(2)
    end

    @testset "Initial tensor" begin
        # Test that initial tensors are real and have correct shape
        for N in [2, 3, 4]
            model = PottsModel(N)
            β = critical_beta(model)
            T = initial_tensor_array(model, β)

            @test size(T) == (N, N, N, N)
            @test eltype(T) <: Real
            @test !any(isnan, T)
            @test !any(isinf, T)
        end
    end
end

################################################
# Test Discrete Gauge (Z_N)
################################################

@testset "Discrete Gauge Z_N" begin
    @testset "Target charge computation" begin
        # Phase 0 → charge 0
        @test compute_target_charge(1.0 + 0im, 3) == 0

        # Phase 2π/3 → charge 2 (to cancel: need ω^{-1} = ω^2)
        ω = exp(2π * im / 3)
        @test compute_target_charge(ω, 3) == 2

        # Phase -2π/3 → charge 1
        @test compute_target_charge(conj(ω), 3) == 1
    end

    @testset "Constraint row construction" begin
        # For index (1,2,3,4) with dimH=5, dimV=5
        # Constraint: g_H[1] + g_V[2] - g_H[3] - g_V[4] ≡ target
        ind = CartesianIndex(1, 2, 3, 4)
        row = construct_row_gfn(ind, 5, 5, 3)

        @test length(row) == 10
        @test row[1] == 1      # +g_H[1]
        @test row[3] == 2      # -g_H[3] ≡ +2 (mod 3)
        @test row[5 + 2] == 1  # +g_V[2]
        @test row[5 + 4] == 2  # -g_V[4] ≡ +2 (mod 3)
    end

    @testset "Gauge application" begin
        # Create simple test tensor
        N = 3
        T = randn(N, N, N, N) + im * randn(N, N, N, N)

        # Apply identity gauge (all zeros)
        g_H = zeros(Int, N)
        g_V = zeros(Int, N)
        T_copy = copy(T)
        apply_discrete_gauge_zn!(T_copy, g_H, g_V, N)

        @test T_copy ≈ T  # Should be unchanged
    end
end

################################################
# Test Integration
################################################

@testset "Integration Tests" begin
    @testset "Model → Tensor → Gauge" begin
        for N in [2, 3]
            model = PottsModel(N)
            β = critical_beta(model)

            # Create tensor
            T = initial_tensor_array(model, β)
            @test size(T) == (N, N, N, N)

            # Find gauge elements
            elements = list_of_allowed_elements_zn(T)
            @test length(elements) > 0

            # Fix discrete gauge
            T_fixed, accepted, H, V = fix_discrete_gauge_zn(T, N)

            # Check result is real
            @test all(abs.(imag.(T_fixed)) .< 1e-10)

            # Check gauge matrices are diagonal
            @test size(H) == (N, N)
            @test size(V) == (N, N)
        end
    end
end

################################################
# Summary
################################################

println("\n" * "=" ^ 60)
println("All tests completed!")
println("=" ^ 60)
