module KS

export KrusellSmith, Approx, Solution, solve_steady, solve_fluctuations, Euler_errors, lm_errors, plot_2dobj, plot_measure, plot_3dobj, stats_crosssection, plot_lorenz, stats_aggregate

using QuantEcon
using CompEcon
using Interpolations
using StatsBase

# ------------------------------------- #
# Define the Krusell Smith economy type #
# ------------------------------------- #

"""
Krusell Smith 1998 deals with an heterogenous agent, general equilibrium model with aggregate fluctuations where households are subject to aggregate-state dependent, uninsurable idiosyncratic risk.
          One model period equals one quarter.

##### Fields
* `β::Float64`: Time discounting parameter
* `σ::Float64`: Risk aversion parameter
* `δ::Float64`: Capital depreciation rate
* `α::Float64`: Capital share of income
* `zg::Float64`: Exogenous aggregate productivity in the good state
* `zb::Float64`: Exogenous aggregate productivity in the bad state
* `zss::Float64`: Aggregate productivity in steady state
* `ug::Float64`: Exogenous unemployment rate in the good state
* `ub::Float64`: Exogenous unemployment rate in the bad state
* `uss::Float64`: Unemployment rate in steady state
* `l̄::Float64`: Time endowment (such that aggregate labor in the bad state is 1.0, i.e. (1 - ub) * l̄ = 1.0)
* `ϵh::Float64`: Exogenous idiosyncratic employment in the high state (= employed)
* `ϵl::Float64`: Exogenous idiosyncratic employment in the low state (= unemployed)
* `Pz::Matrix{Float64}` : Transition matrix for exogenous aggregate state (state 1 is bad, state 2 is good)
* `Pϵ_gg::Matrix{Float64}` : Transition matrix for exogenous idiosyncratic state, given aggregate state transition from good to good (state 1 is low, state 2 is high)
* `Pϵ_bb::Matrix{Float64}` : [as above] from bad to bad
* `Pϵ_bg::Matrix{Float64}` : [as above] from bad to good
* `Pϵ_gb::Matrix{Float64}` : [as above] from good to bad
* `Pϵ::Tuple{Tuple{Matrix{Float64}, Matrix{Float64}}, Tuple{Matrix{Float64}, Matrix{Float64}}}` : Tuple of tuples that combines the previous transition matrices for exogenous idiosyncratic states. The first tuple layer refers to the current exogenous aggregate state s (1 is bad state, 2 is good state); the second tuple refers to the next period exogenous aggregate state s', conditional on s. For instance, Pϵ_bg would be (Pϵ[1])[2]
* `Pϵss::Matrix{Float64}` : Transition matrix for exogenous idiosyncratic state in steady state
"""
immutable KrusellSmith
    β::Float64
    σ::Float64
    δ::Float64
    α::Float64
    zg::Float64
    zb::Float64
    zss::Float64
    ug::Float64
    ub::Float64
    uss::Float64
    l̄::Float64
    ϵh::Float64
    ϵl::Float64
    Pz::Matrix{Float64}
    Pϵ_gg::Matrix{Float64}
    Pϵ_bb::Matrix{Float64}
    Pϵ_bg::Matrix{Float64}
    Pϵ_gb::Matrix{Float64}
    Pϵ::Tuple{Tuple{Matrix{Float64}, Matrix{Float64}}, Tuple{Matrix{Float64}, Matrix{Float64}}}
    Pϵss::Matrix{Float64}
end

function Base.show(io::IO, m::KrusellSmith)
    @printf io "Household Parameters\n"
    @printf io "--------------------\n"
    @printf io " β:      %1.3f\n" m.β
    @printf io " σ:      %1.3f\n" m.σ
    @printf io " l̄:      %1.3f\n" m.l̄
    @printf io "\n"
    @printf io "Production Parameters\n"
    @printf io "---------------------\n"
    @printf io " δ:      %1.3f\n" m.δ
    @printf io " α:      %1.3f\n" m.α
    @printf io "\n"
    @printf io "Aggregate State Process\n"
    @printf io "-----------------------\n"
    @printf io " P[s]:\n"
    @printf io " | %1.3f %1.3f |\n" m.Pz[1, 1] m.Pz[1, 2]
    @printf io " | %1.3f %1.3f |\n" m.Pz[2, 1] m.Pz[2, 2]
    @printf io "\n"
    @printf io "Productivity\n"
    @printf io "------------\n"
    @printf io " z(b):  %1.3f\n" m.zb
    @printf io " z(g):  %1.3f\n" m.zg
    @printf io " E[z]:  %1.3f\n" m.zss
    @printf io "\n"
    @printf io "Unemployment Rate\n"
    @printf io "-----------------\n"
    @printf io " u(b):  %1.3f\n" m.ub
    @printf io " u(g):  %1.3f\n" m.ug
    @printf io " E[u]:  %1.3f\n" m.uss
    @printf io "\n"
    @printf io "Idiosyncratic State Process\n"
    @printf io "---------------------------\n"
    @printf io " P[ϵ|s = b, s' = b]:\n"
    @printf io " | %1.3f %1.3f |\n" m.Pϵ_bb[1, 1] m.Pϵ_bb[1, 2]
    @printf io " | %1.3f %1.3f |\n" m.Pϵ_bb[2, 1] m.Pϵ_bb[2, 2]
    @printf io "\n"
    @printf io " P[ϵ|s = b, s' = g]:\n"
    @printf io " | %1.3f %1.3f |\n" m.Pϵ_bg[1, 1] m.Pϵ_bg[1, 2]
    @printf io " | %1.3f %1.3f |\n" m.Pϵ_bg[2, 1] m.Pϵ_bg[2, 2]
    @printf io "\n"
    @printf io " P[ϵ|s = g, s' = b]:\n"
    @printf io " | %1.3f %1.3f |\n" m.Pϵ_gb[1, 1] m.Pϵ_gb[1, 2]
    @printf io " | %1.3f %1.3f |\n" m.Pϵ_gb[2, 1] m.Pϵ_gb[2, 2]
    @printf io "\n"
    @printf io " P[ϵ|s = g, s' = g]:\n"
    @printf io " | %1.3f %1.3f |\n" m.Pϵ_gg[1, 1] m.Pϵ_gg[1, 2]
    @printf io " | %1.3f %1.3f |\n" m.Pϵ_gg[2, 1] m.Pϵ_gg[2, 2]
    @printf io "\n"
    @printf io " P[ϵ] in steady state:\n"
    @printf io " | %1.3f %1.3f |\n" m.Pϵss[1, 1] m.Pϵss[1, 2]
    @printf io " | %1.3f %1.3f |\n" m.Pϵss[2, 1] m.Pϵss[2, 2]
    @printf io "\n"
end

"""
This is the constructor for building a Krusell Smith economy as presented in Krusell Smith 1998

##### Arguments
* `;β::Float64(0.99)`: Time discounting parameter
* `;σ::Float64(1.0)`: Risk aversion parameter
* `;δ::Float64(0.025)`: Capital depreciation rate
* `;α::Float64(0.36)`: Capital share of income
* `;zg::Float64(1.01)`: Exogenous aggregate productivity in the good state
* `;zb::Float64(0.99)`: Exogenous aggregate productivity in the bad state
* `;ug::Float64(0.04)`: Exogenous unemployment rate in the good state
* `;ub::Float64(0.1)`: Exogenous unemployment rate in the bad state
* `;ϵh::Float64(1.0)`: Exogenous idiosyncratic employment in the high state
* `;ϵl::Float64(0.0)`: Exogenous idiosyncratic employment in the low state

##### Returns
* `x::KrusellSmith`: Instance of Krusell Smith economy
"""
function KrusellSmith(;β = 0.99, σ = 1.0, δ = 0.025, α = 0.36, zg = 1.01, zb = 0.99, ug = 0.04, ub = 0.1, ϵh = 1.0, ϵl = 0.0)
    # Test admissible parameter restrictions:
    (β < 1.0) && (β >= 0.0) ? nothing : error("Time discounting parameter β must be in [0, 1)")
    σ > 0 ? nothing : error("Risk aversion parameter σ must be > 0")
    (δ <= 1.0) && (δ >= 0.0) ? nothing : error("Capital depreciation rate δ must be in [0, 1]")
    (α <= 1.0) && (α >= 0.0) ? nothing : error("Capital share of income α must be in [0, 1]")
    zb < zg ? nothing : error("Aggregate productivity in bad state must be lower than in good state")
    ub > ug ? nothing : error("Unemployment rate in bad state must be higher than in good state")

    # Compute time endowment:
    l̄ = 1.0 / (1 - ub)

    # Infer transition probabilities given restrictions:
    # 1: π_ss'00 + π_ss'01 = π_ss'10 + π_ss'11 = π_ss' for all (s, s')
    # 2: u_s * π_ss'00/π_ss' + (1 - u_s) * π_ss'01/π_ss' = u_s' for all (s, s')
    # 3: π_gb00/π_gb = 1.25 * π_bb00/π_bb
    # 4: π_bg00/π_bg = 0.75 * π_gg00/π_gg
    # 5: 1/(1 - π_gg00/π_gg) = 1.5
    # 6: 1/(1 - π_bb00/π_bb) = 2.5
    # 7: 1/(1 - π_gg) = 8
    # 8: 1/(1 - π_bb) = 8
    # 9: π_gb = 1 - π_gg
    # 10: π_bg = 1 - π_bb

    π_gg = π_bb = 7/8
    Pz = [π_bb 1-π_bb; 1-π_gg π_gg]

    # Let
    # [x1 x2 x3 x4   [π_gg00/π_gg π_gg01/π_gg π_gg10/π_gg π_gg11/π_gg
    #  y1 y2 y3 y4 =  π_bb00/π_bb π_bb01/π_bb π_bb10/π_bb π_bb11/π_bb
    #  z1 z2 z3 z4    π_bg00/π_bg π_bg01/π_bg π_bg10/π_bg π_bg11/π_bg
    #  w1 w2 w3 w4]   π_gb00/π_gb π_gb01/π_gb π_gb10/π_gb π_gb11/π_gb]

    # Solve the system:
    x1 = 0.5/1.5
    y1 = 1.5/2.5
    z1 = 0.75 * x1
    w1 = 1.25 * y1

    col1 = [x1; y1; z1; w1]
    col2 = 1 - col1
    col3 = ([ug; ub; ug; ub] - [ug; ub; ub; ug] .* col1)./(1 - [ug; ub; ub; ug])
    col4 = 1 - col3
    sol = [col1'' col2'' col3'' col4''] .* [π_gg; π_bb; 1-π_bb; 1-π_gg]''

    # Complete assigning transition matrices:
    condprobs(P::Matrix{Float64}) = P./sum(P, 2)
    function reshape_sol(x::Array{Float64})
        # reshape vector to matrix with correct transitions (see comments above)
        P = reshape(x, 2, 2)'
        # fill in with conditional probabilities
        condprobs(P)
    end
    Pϵ_gg, Pϵ_bb, Pϵ_bg, Pϵ_gb = reshape_sol(sol[1, :]), reshape_sol(sol[2, :]), reshape_sol(sol[3, :]), reshape_sol(sol[4, :])

    # Determine steady state elements:
    πz =  mc_compute_stationary(MarkovChain(Pz))
    zss = dot(πz, [zb; zg])
    uss = dot(πz, [ub; ug])

    Pzϵ = condprobs([Pϵ_bb Pϵ_bg; Pϵ_gb Pϵ_gg])
    πb0, πb1, πg0, πg1 = mc_compute_stationary(MarkovChain(Pzϵ))
    Pϵss = condprobs([πg0 πg0; πg1 πg1] .* (Pϵ_gg + Pϵ_gb) + [πb0 πb0; πb1 πb1] .* (Pϵ_bb + Pϵ_bg))

    # Test transition matrices:
    function test_P(P::Matrix{Float64})
        (sum(P .< 0.0) == 0) && (sum(P .> 1.0) == 0) ? nothing : error("Transition matrix elemenents must be in [0, 1]")
        size(P)[1] == size(P)[2] ? nothing : error("Transition matrix must be square")
        maxabs(sum(P, 2) - ones(size(P)[1])) < eps() ? nothing : error("Rows of transition matrix must sum to 1")
    end
    test_P(Pz)
    test_P(Pϵ_gg)
    test_P(Pϵ_bb)
    test_P(Pϵ_bg)
    test_P(Pϵ_gb)
    test_P(Pϵss)

    # combine transition matrices of ϵ in tuples:
    Pϵ_bsprime = (Pϵ_bb, Pϵ_bg)
    Pϵ_gsprime = (Pϵ_gb, Pϵ_gg)
    Pϵ = (Pϵ_bsprime, Pϵ_gsprime)

    KrusellSmith(β, σ, δ, α, zg, zb, zss, ug, ub, uss, l̄, ϵh, ϵl, Pz, Pϵ_gg, Pϵ_bb, Pϵ_bg, Pϵ_gb, Pϵ, Pϵss)
end

# ----------------------------- #
# Define the approximation type #
# ----------------------------- #

"""
This is the type holding the parameters and elements of the solution algorithm to the Krusell Smith economy

##### Fields
* `nk::Int64`: Number of knots to use in approximation with respect to capital holding
* `nkf::Int64`: Number of knots for capital holding to use in approximation of stationary density with GTH algorithm from QuantEcon (nkf > nk)
* `nϵ::Int64`: Number of idiosyncratic employment states
* `nK::Int64`: Number of knots to use in approximation with respect to aggregate capital
* `nz::Int64`: Number of aggregate exogenous states
* `kmin::Float64`: Lower bound on capital holding (because household's labor income is 0.0 with positive probability, a borrowing limit never binds in this model)
* `kmax::Float64`: Upper bound on capital holding
* `Kmin::Float64`: Lower bound on aggregate capital
* `Kmax::Float64`: Upper bound on aggregate capital
* `curv::Float64`: Parameter for transformation of evenly-spaced knots to concentrate them near the constraint
* `gridk::Vector{Float64}`: Grid on capital holding space
* `gridkf::Vector{Float64}`: Finer grid on capital holding space for stationary density
* `bkf::Basis{1}`: CompEcon piecewise-linear interpoland parameters with finer grid on capital holding space
* `gridϵ::Vector{Float64}`: Idiosyncratic employment state vector (already multiplied by the time endowment l̄)
* `cartk::Matrix{Float64}`: Capital holding values from the Cartesian product {knotsk} * {ϵl, ϵh}
* `cartkf::Matrix{Float64}`: Capital holding values from the Cartesian product {knotskf} * {ϵl, ϵh} with finer grid
* `cartϵ::Matrix{Float64}`: Employment state values from the Cartesian product {knotsk} * {ϵl, ϵh} (already multiplied by the time endowment)
* `gridk::Vector{Float64}`: Grid on aggregate capital space
* `gridz::Vector{Float64}`: Exogenous aggregate state vector
* `gridL::Vector{Float64}`: Aggregate labor vector (generic element is (1 - u_s) * l̄ where s is the exogenous aggregate state)
"""
immutable Approx
    nk::Int64
    nkf::Int64
    nϵ::Int64
    nK::Int64
    nz::Int64
    kmin::Float64
    kmax::Float64
    curv::Float64
    Kmin::Float64
    Kmax::Float64
    gridk::Vector{Float64}
    gridkf::Vector{Float64}
    bkf::Basis{1}
    gridϵ::Vector{Float64}
    cartk::Matrix{Float64}
    cartkf::Matrix{Float64}
    cartϵ::Matrix{Float64}
    gridK::Vector{Float64}
    gridz::Vector{Float64}
    gridL::Vector{Float64}
end

function Base.show(io::IO, a::Approx)
    @printf io "Grids\n"
    @printf io "-----\n"
    @printf io " un-evenly spaced for capital holding k: %d points in [%1.2f, %1.2f], curvature %1.2f\n" a.nk a.kmin a.kmax a.curv
    @printf io "\n"
    @printf io " un-evenly spaced for capital holding k, finer (density approximation): %d points in [%1.2f, %1.2f], curvature %1.2f\n" a.nkf a.kmin a.kmax a.curv
    @printf io "\n"
    @printf io " evenly spaced for aggregate capital K: %d points in [%1.2f, %1.2f]\n" a.nK a.Kmin a.Kmax
    @printf io "\n"
    @printf io " %d idiosyncratic labor values l̄ϵ: {%1.1f, %1.1f}\n" a.nϵ a.gridϵ[1] a.gridϵ[2]
    @printf io "\n"
    @printf io " %d aggregate productivity values z: {%1.2f, %1.2f}\n" a.nz a.gridz[1] a.gridz[2]
    @printf io "\n"
    @printf io " %d aggregate labor values L: {%1.2f, %1.2f}\n" a.nz a.gridL[1] a.gridL[2]
end

"""
This is the constructor for building the approximation elements used by the solution algorithm

##### Arguments
* `;nk::Int64(125)`: Number of knots to use in approximation with respect to capital holding
* `;nkf::Int64(2 * nk)`: Number of knots for capital holding to use in approximation of stationary density with GTH algorithm from QuantEcon (nkf > nk)
* `;nK::Int64(15)`: Number of knots to use in approximation with respect to aggregate capital
* `;kmin::Float64(1e-4)`: Lower bound on capital holding
* `;kmax::Float64(175.0)`: Upper bound on capital holding
* `;curv::Float64(0.4)`: Parameter for transformation of evenly-spaced knots to concentrate them near the constraint
* `;Kmin::Float64(35.0)`: Lower bound on aggregate capital
* `;Kmax::Float64(44.0)`: Upper bound on aggregate capital

##### Returns
* `x::Approx`: Instance of approximation elements for the Krusell Smith economy
"""
function Approx(; nk = 125, nkf = 2 * nk, nK = 15, kmin = 1e-4, kmax = 175.0, curv = 0.4, Kmin = 35.0, Kmax = 44.0)
    m = KrusellSmith()
    zb, zg, ub, ug, l̄, ϵl, ϵh = m.zb, m.zg, m.ub, m.ug, m.l̄, m.ϵl, m.ϵh

    gridk = max(min((linspace(0.0, (kmax - kmin)^curv, nk)).^(1 / curv) + kmin, kmax), kmin)
    gridkf = max(min((linspace(0.0, (kmax - kmin)^curv, nkf)).^(1 / curv) + kmin, kmax), kmin)
    bkf = Basis(LinParams(gridkf, 0))
    gridϵ = [ϵl; ϵh] * l̄
    nϵ = length(gridϵ)
    cartk = repmat(gridk'', 1, nϵ)
    cartkf = repmat(gridkf'', 1, nϵ)
    cartϵ = repmat(gridϵ', nk, 1)

    gridK = collect(linspace(Kmin, Kmax, nK))
    gridz = [zb; zg]
    nz = length(gridz)
    gridL = [1 - ub; 1 - ug] * l̄

    Approx(nk, nkf, nϵ, nK, nz, kmin, kmax, curv, Kmin, Kmax, gridk, gridkf, bkf, gridϵ, cartk, cartkf, cartϵ, gridK, gridz, gridL)
end

# ------------------------------------- #
# Define the approximate solution types #
# ------------------------------------- #

"""
This is the type holding the approximate steady-state solution elements

##### Fields
* `r::Float64`: Steady state net interest rate
* `K::Float64`: Steady state aggregate capital
* `polk::Matrix{Float64}`: Steady state capital-holding policy function for a household
* `polc::Matrix{Float64}`: Steady state consumption policy function for a household
* `vf::Matrix{Float64}`: Steady state value function for a household
* `μ::Matrix{Float64}`: Steady state stationary density on idiosyncratic states
"""
type SteadySolution
    r::Float64
    K::Float64
    polk::Matrix{Float64}
    polc::Matrix{Float64}
    vf::Matrix{Float64}
    μ::Matrix{Float64}
end

function Base.show(io::IO, ss::SteadySolution)
    @printf io "Steady State Equilibrium Aggregates\n"
    @printf io "-----------------------------------\n"
    @printf io " Interest rate: %1.2f percent\n" ss.r * 100
    @printf io " Capital: %1.2f\n" ss.K
end

"""
This is the type holding the approximate solution to the economy with aggregate fluctuations

##### Fields
* `lm::Tuple{Vector{Float64}, Vector{Float64}}`: Tuple that holds the parameters of the law of motion of aggregate capital (this refers to the quasi-aggregation result of Krusell ans Smith 1998). The first element in the tuple yields K' as a function of K and current exogenous aggregate state 1 (bad state); the second element corresponds to state 2 (good state)
* `polk::Array{Float64, 4}`: Capital-holding policy function for a household in the economy with aggregate fluctuations
* `polc::Array{Float64, 4}`: Consumption policy function for a household in the economy with aggregate fluctuations
"""
type AggregateSolution
    lm::Tuple{Vector{Float64}, Vector{Float64}}
    polk::Array{Float64, 4}
    polc::Array{Float64, 4}
end

function Base.show(io::IO, fl::AggregateSolution)
    @printf io "Approximate Law of Motion\n"
    @printf io "-------------------------\n"
    @printf io " if s = b: log(K') = %1.4f + %1.4f log(K)\n" (fl.lm[1])[1] (fl.lm[1])[2]
    @printf io " if s = g: log(K') = %1.4f + %1.4f log(K)\n" (fl.lm[2])[1] (fl.lm[2])[2]
end

# ------------------------------------------------------- #
# Define primitive functions of the Krusell Smith economy #
# ------------------------------------------------------- #

"""
CRRA utility function

##### Arguments
* `m::KrusellSmith`: Object holding the parameter values of the Krusell Smith economy
* `c::Array{Float64}`: Array of consumption values

##### Returns
* `x::Array{Float64}`: Array of utility values
"""
function U(m::KrusellSmith, c::Array{Float64})
    σ = m.σ
    if σ == 1.0
        return log(c)
    else
        return (1 - σ)^(- 1) * (c.^(1 - σ) - 1)./(1 - σ)
    end
end

"""
Net interest rate function

##### Arguments
* `m::KrusellSmith`: Object holding the parameter values of the Krusell Smith economy
* `z::Float64`: Exogenous aggregate productivity
* `K::Float64`: Aggregate capital
* `L::Float64`: Aggregate labor

##### Returns
* `x::Float64`: Net interest rate
"""
function irate(m::KrusellSmith, z::Float64, K::Float64, L::Float64)
    α, δ = m.α, m.δ
    α * z * (L / K)^(1 - α) - δ
end

"""
Wage function

##### Arguments
* `m::KrusellSmith`: Object holding the parameter values of the Krusell Smith economy
* `z::Float64`: Exogenous aggregate productivity
* `R::Float64`: Gross interest rate

##### Returns
* `x::Float64`: Wage
"""
function wage(m::KrusellSmith, z::Float64, R::Float64)
    α, δ = m.α, m.δ
    (1 - α) * (α / (R - 1 + δ))^(α / (1-α)) * z^(1 / (1 - α))
end

# ---------------------------------#
# Stochastic steady state solution #
# -------------------------------- #

"""
This function solves for the approximate household's policy functions in the stochastic steady state using the Endogenous Grid Method (EGM)

##### Arguments
* `m::KrusellSmith`: Object holding the parameter values of the Krusell Smith economy
* `a::Approx`: Object holding the approximation elements used in the solution algorithm
* `R::Float64`: Steady state gross interest rate
* `;maxiter::Int64(10000)`: Maximum number of iterations for the EGM algorithm
* `;tol::Float64(1e-10)`: Tolerance to stop the EGM algorithm
* `;verbose::Bool(false)`: true = print text; false = do not print text

##### Returns
* `polk::Matrix{Float64}`: Steady state capital-holding policy function for a household
* `polc::Matrix{Float64}`: Steady state consumption policy function for a household
"""
function solve_policies(m::KrusellSmith, a::Approx, R::Float64; maxiter = 10000, tol = 1e-10, verbose = false)
    R >= 1 - m.δ ? nothing : error("Interest rate in household's problem solution must be gross interest rate")

    β, σ, zss, Pϵss = m.β, m.σ, m.zss, m.Pϵss
    nk, nϵ, kmin, kmax, gridk, gridϵ, cartk, cartϵ = a.nk, a.nϵ, a.kmin, a.kmax, a.gridk, a.gridϵ, a.cartk, a.cartϵ
    w = wage(m, zss, R)

    # initial guess for consumption policy function
    polc = (R - 1) * cartk + w * cartϵ

    polc_next = similar(polc)

    for n in 1:maxiter
        # given the guess, invert the Euler equation to obtain ̂c(k',ϵ)
        ĉ = (β * R * polc.^(- σ) * Pϵss').^(- 1 / σ)
        # use the budget constraint to obtain k(k',ϵ) (this is the endogenous grid)
        endGRIDk = 1 / R * (ĉ + cartk - w * cartϵ)

        # interpolate ̂c on endGRIDk
        for iϵ in 1:nϵ, ik in 1:nk
            # take the endogenous grid at each employment level
            endgridk = endGRIDk[:, iϵ]
            # pull out k(k' = kmin,ϵ)
            endgridk_min = endgridk[1]
            itp_ĉ = interpolate((endgridk,), ĉ[:, iϵ], Gridded(Linear()))

            k, ϵ = gridk[ik], gridϵ[iϵ]
            if k < endgridk_min # if the capital holding level is below the minimum level which makes the constraint bind, then the constraint necessarily binds and consumption comes from the budget constraint
                polc_next[ik, iϵ] = R * k + w * ϵ - kmin
            else
                polc_next[ik, iϵ] = itp_ĉ[k]
            end
        end

        err = maxabs(polc_next - polc)
        copy!(polc, polc_next)
        if (n % 50 == 0) && verbose; @printf("Iter. # %d, Dist. = %1.2e\n", n, err); end
        if n == maxiter; warn("Household's policy function in steady state: hit maxiter"); end

        if err < tol; break; end
    end

    # get capital-holding policy function
    polk = R * cartk + w * cartϵ - polc
    # if maximum(polk) > kmax; warn("Household's capital-holding policy function exceeds kmax"); end

    polk, polc
end

"""
This function solves for the approximate household's value function in the stochastic steady state

##### Arguments
* `m::KrusellSmith`: Object holding the parameter values of the Krusell Smith economy
* `a::Approx`: Object holding the approximation elements used in the solution algorithm
* `polk::Vector{Float64}`: Steady state capital-holding policy function for a household
* `polc::Vector{Float64}`: Steady state consumption policy function for a household
* `;maxiter::Int64(10000)`: Maximum number of iterations on the contraction operator
* `;tol::Float64(1e-10)`: Tolerance to stop the contraction algorithm

##### Returns
* `vf::Matrix{Float64}`: Steady state value function for a household
"""
function solve_vf(m::KrusellSmith, a::Approx, polk::Matrix{Float64}, polc::Matrix{Float64}; maxiter = 10000, tol = 1e-10)
    β, Pϵss = m.β, m.Pϵss
    nk, nϵ, gridk = a.nk, a.nϵ, a.gridk

    # vector of flow utility
    uflow = U(m, polc)

    vf, cvf = zeros(nk, nϵ), similar(polc)

    for n in 1:maxiter
        for iϵ in 1:nϵ
            # get continuation value by interpolating value function, as k' is not on the grid
            itp_cvf = interpolate((gridk, ), vf[:,iϵ], Gridded(Linear()))
            cvf[:,iϵ] = itp_cvf[polk[:,iϵ]]
        end

        # iteration of contraction operator
        vf_next = uflow + β * (cvf * Pϵss')

        err = maxabs(vf_next - vf)
        copy!(vf, vf_next)
        if n == maxiter; warn("Household's value function in steady state: hit maxiter"); end

        if err < tol; break; end
    end

    vf
end

"""
This function solves for the approximate stationary density on a finer grid on idiosyncratic states using the GTH algorithm from QuantEcon

##### Arguments
* `m::KrusellSmith`: Object holding the parameter values of the Krusell Smith economy
* `a::Approx`: Object holding the approximation elements used in the solution algorithm
* `polk::Vector{Float64}`: Steady state capital-holding policy function for a household

##### Returns
* `x::Matrix{Float64}`: Steady state stationary density on idiosyncratic states
"""
function solve_density(m::KrusellSmith, a::Approx, polk::Matrix{Float64})
    Pϵss = m.Pϵss
    kmax, nkf, nϵ, gridk, gridkf, bkf = a.kmax, a.nkf, a.nϵ, a.gridk, a.gridkf, a.bkf

    # compute capital holding decision on the finer Cartesian product
    kprime = Array(Float64, nkf, nϵ)
    for iϵ in 1:nϵ
        itp_polk = interpolate((gridk, ), polk[:, iϵ], Gridded(Linear()))
        kprime[:, iϵ] = itp_polk[gridkf]
    end
    # vectorize, so for instance kprime[1:2] = [itp_polk[gridkf[1], 1]; itp_polk[gridkf[2], 1]] (knots for first state change faster)
    # cap extrapolation at kmax to have well-defined transition
    kprime = min(vec(kprime), kmax)

    # use CompEcon basis matrix to compute transition matrix for endogenous state
	Qk = BasisStructure(bkf, Direct(), kprime).vals[1]
    # extend transition matrix Pϵss to match the first state changing faster (see comment above)
    longPϵss = kron(Pϵss, ones(nkf, 1))
    # combine endogenous and exogenous transition
    Q = row_kron(longPϵss, Qk)

    # find stationary density and reshape to size [nkf, nϵ]
    μ = mc_compute_stationary(MarkovChain(Q))
    reshape(μ, nkf, nϵ)
end

"""
This function solves for the stochastic steady state of the economy using bisection on the interest rate

##### Arguments
* `m::KrusellSmith`: Object holding the parameter values of the Krusell Smith economy
* `a::Approx`: Object holding the approximation elements used in the solution algorithm
* `;maxiter::Int64(30)`: Maximum number of iterations for bisection
* `;tol::Float64(1e-8)`: Tolerance to stop bisection
* `;w_hh::Float64(0.5)`: Weight on interest rate from household's problem when updating to new interest rate
* `;lb::Float64(0.0)`: Inital lower bound on net interest rate for bisection
* `;ub::Float64(1 / m.β - 1)`: Inital upper bound on net interest rate for bisection
* `;verbose::Bool(true)`: true = print text; false = do not print text

##### Returns
* `x::SteadySolution`: Approximate steady-state solution elements
"""
function solve_steady(m::KrusellSmith, a::Approx; maxiter = 30, tol = 1e-8, w_hh = 0.5, lb = 0.0, ub = (1 / m.β - 1), verbose = true)
    δ, α, zss, uss, l̄ = m.δ, m.α, m.zss, m.uss, m.l̄
    nϵ, gridkf, cartk, cartkf = a.nϵ, a.gridkf, a.cartk, a.cartkf

    polk, polc, μ, K_s = similar(cartk), similar(cartk), similar(cartkf), 0.0

    for n in 1:maxiter
        width = ub - lb

        # net interest rate faced by the household
        r_hh = 0.5 * (lb + ub)

        # solve the household's problem
        polk, polc = solve_policies(m, a, 1 + r_hh)
        # solve for the stationary density
        μ = solve_density(m, a, polk)

        # compute aggregate variables
        K_s = sum(cartkf .* μ)
        L = l̄ * (1 - uss)

        # net interest rate from firm's optimization and capital demand
        r_firm = irate(m, zss, K_s, L)
        K_d = (α * zss / (r_hh + δ))^(1 / (1 - α)) * L

        if verbose
            @printf("Iter. # %d, Width = %1.2e, Interest rate = %1.7f\n", n, width, r_hh)
            @printf(" -- Capital supply = %1.4f, Capital demand = %1.4f\n", K_s, K_d)
            @printf("\n")
        end
        if n == maxiter; warn("Interest rate in steady state: hit maxiter"); end

        if width < tol
            break
        else
            # update
        	r_upd = w_hh * r_hh + (1 - w_hh) * r_firm
        	r_upd > r_hh ? lb = r_hh: ub = r_hh
        end
    end

    # solve value function at the equilibrium
    vf = solve_vf(m, a, polk, polc)

    @printf("Steady state done!\n")
    SteadySolution(0.5 * (lb + ub), K_s, polk, polc, vf, μ)
end

# --------------------------------------------------- #
# Solution of the economy with aggregate fluctuations #
# --------------------------------------------------- #

"""
This function solves for the approximate household's policy functions in the economy with aggregate fluctuations using the EGM

##### Arguments
* `m::KrusellSmith`: Object holding the parameter values of the Krusell Smith economy
* `a::Approx`: Object holding the approximation elements used in the solution algorithm
* `lm::Tuple{Vector{Float64}, Vector{Float64}}`: Tuple that holds the parameters of the law of motion of aggregate capital (this refers to the quasi-aggregation result of Krusell ans Smith 1998). The first element in the tuple yields K' as a function of K and current exogenous aggregate state 1 (bad state); the second element corresponds to state 2 (good state)
* `;maxiter::Int64(10000)`: Maximum number of iterations for the EGM algorithm
* `;tol::Float64(1e-10)`: Tolerance to stop the EGM algorithm
* `;verbose::Bool(false)`: true = print text; false = do not print text

##### Returns
* `polk::Array{Float64, 4}`: Capital-holding policy function for a household in the economy with aggregate fluctuations
* `polc::Array{Float64, 4}`: Consumption policy function for a household in the economy with aggregate fluctuations
"""
function solve_policies(m::KrusellSmith, a::Approx, lm::Tuple{Vector{Float64}, Vector{Float64}}; maxiter = 10000, tol = 1e-10, verbose = false)
    β, σ, Pz, Pϵ = m.β, m.σ, m.Pz, m.Pϵ
    nk, nϵ, nK, nz, kmin, kmax, gridk, gridϵ, cartk, cartϵ, gridK, gridz, gridL = a.nk, a.nϵ, a.nK, a.nz, a.kmin, a.kmax, a.gridk, a.gridϵ, a.cartk, a.cartϵ, a.gridK, a.gridz, a.gridL

    # initial guess for consumption policy function (standard guess, at r = .005 and z = 1.0)
    polc = repeat(0.005 * cartk + wage(m, 1.0, 1.005) * cartϵ, inner = [1, 1, nK, nz])

    polc_next = similar(polc)

    # given law of motion, determine K' on the Cartesian product {knotsK} * {zb, zg}
    cartKprime = Array(Float64, nK, nz)
    regK = [ones(nK, 1) log(gridK'')]
    for iz in 1:nz
        cartKprime[:, iz] = exp(sum(repmat(lm[iz]', nK, 1) .* regK, 2))
    end

    for n in 1:maxiter
        # since polc is defined on gridK, but K' values are not on the grid - and depend on z -, define c' on K'(K, z) through interpolation
        itp_polc = interpolate((1:nk, 1:nϵ, gridK, 1:nz), polc, (NoInterp(), NoInterp(), Gridded(Linear()), NoInterp()))
        # c' depends on K', which in turns depends on current z, so create 2 arrays cprime_s where s is either b or g
        cprime_b = similar(polc)
        cprime_g = similar(polc)
        for I in CartesianRange((nk, nϵ, nK, nz))
            for iz in 1:nz
                ikprime, iϵprime, Kprime, izprime = I[1], I[2], cartKprime[I[3], iz], I[4]
                if iz == 1
                    cprime_b[I] = itp_polc[ikprime, iϵprime, Kprime, izprime]
                elseif iz == 2
                    cprime_g[I] = itp_polc[ikprime, iϵprime, Kprime, izprime]
                end
            end
        end
        # tuple that holds c' depending on current exogenous aggregate state (1 is bad, 2 is good)
        cprime = (cprime_b, cprime_g)

        ĉ = Array(Float64, nk, nϵ, nK, nz)
        endGRIDk = similar(ĉ)
        for iz in 1:nz, iK in 1:nK
            margU = zeros(nk, nϵ)
            z, Kprime = gridz[iz], cartKprime[iK, iz]
            # get expected marginal utility
            for izprime in 1:nz
                Rprime = irate(m, gridz[izprime], Kprime, gridL[izprime]) + 1
                margU += Pz[iz, izprime] * β * Rprime * ((cprime[iz])[:, :, iK, izprime].^(- σ) * (Pϵ[iz])[izprime]')
            end
            # given the guess, invert the Euler equation to obtain ̂c(k',ϵ,K,z)
            holdĉ = margU.^(- 1 / σ)
            ĉ[:, :, iK, iz] = holdĉ
            # use the budget constraint to obtain k(k',ϵ,K,z) (this is the endogenous grid)
            R = irate(m, z, gridK[iK], gridL[iz]) + 1
            w = wage(m, z, R)
            endGRIDk[:, :, iK, iz] = 1 / R * (holdĉ + cartk - w * cartϵ)
        end

        # interpolate ̂c on endGRIDk
        for iz in 1:nz, iK in 1:nK, iϵ in 1:nϵ, ik in 1:nk
            # take the endogenous grid at each (ϵ,K,z)
            endgridk = endGRIDk[:, iϵ, iK, iz]
            # pull out k(k' = kmin,ϵ,K,z)
            endgridk_min = endgridk[1]
            itp_ĉ = interpolate((endgridk, ), ĉ[:, iϵ, iK, iz], Gridded(Linear()))

            k = gridk[ik]
            if k < endgridk_min # if the capital holding level is below the minimum level which makes the constraint bind, then the constraint necessarily binds and consumption comes from the budget constraint
                ϵ, K, z, L =  gridϵ[iϵ], gridK[iK], gridz[iz], gridL[iz]
                R = irate(m, z, K, L) + 1
                w = wage(m, z, R)
                polc_next[ik, iϵ, iK, iz] = R * k + w * ϵ - kmin
            else
                polc_next[ik, iϵ, iK, iz] = itp_ĉ[k]
            end
        end

        err = maxabs(polc_next - polc)
        copy!(polc, polc_next)
        if (n % 50 == 0) && verbose; @printf("Iter. # %d, Dist. = %1.2e\n", n, err); end
        if n == maxiter; warn("Household's policy function in steady state: hit maxiter"); end

        if err < tol; break; end
    end

    # get capital-holding policy function
    polk = similar(polc)
    for iz in 1:nz, iK in 1:nK
        K, z, L =  gridK[iK], gridz[iz], gridL[iz]
        R = irate(m, z, K, L) + 1
        w = wage(m, z, R)
        polk[:, :, iK, iz] = R * cartk + w * cartϵ - polc[:, :, iK, iz]
    end
    # if maximum(polk) > kmax; warn("Household's capital-holding policy function exceeds kmax"); end

    polk, polc
end

"""
This function updates the approximate density on idiosyncratic states between two periods in the economy with aggregate fluctuations

##### Arguments
* `m::KrusellSmith`: Object holding the parameter values of the Krusell Smith economy
* `a::Approx`: Object holding the approximation elements used in the solution algorithm
* `μ::Vector{Float64}`: Current density on idiosyncratic states (vectorized over points in the Cartesion product {knotskf} * {ϵl, ϵh})
* `K::Float64`: Current aggregate capital
* `iz::Int64`: Current exogenous aggregate state
* `itp_polk::Interpolations.GriddedInterpolation`: Interpolation object for the capital-holding policy function in the economy with aggregate fluctuations
* `Pϵ_ssprime::Matrix{Float64}`: Transition matrix for exogenous idiosyncratic state between current and next period's exogenous aggregate states

##### Returns
* `x::Matrix{Float64}`: Updated density on idiosyncratic states
"""
function update_density(m::KrusellSmith, a::Approx, μ::Vector{Float64}, K::Float64, iz::Int64, itp_polk::Interpolations.GriddedInterpolation, Pϵ_ssprime::Matrix{Float64})
    kmax, nkf, nϵ, gridkf, bkf = a.kmax, a.nkf, a.nϵ, a.gridkf, a.bkf

    # compute capital holding decision on the finer Cartesian product
    kprime = Array(Float64, nkf, nϵ)
    for iϵ in 1:nϵ, ikf in 1:nkf
        kf = gridkf[ikf]
        kprime[ikf, iϵ] = itp_polk[kf, iϵ, K, iz]
    end
    # vectorize, so for instance kprime[1:2] = [itp_polk[gridkf[1], 1]; itp_polk[gridkf[2], 1]] (knots for first state change faster)
    # cap extrapolation at kmax to have well-defined transition
    kprime = min(vec(kprime), kmax)

    # use CompEcon basis matrix to compute transition matrix for endogenous state
	Qk = BasisStructure(bkf, Direct(), kprime).vals[1]
    # extend transition matrix Pϵ_ssprime to match the first state changing faster
    longPϵ_ssprime = kron(Pϵ_ssprime, ones(nkf, 1))
    # combine endogenous and exogenous transition
    Q = row_kron(longPϵ_ssprime, Qk)

    # update density
    Q' * μ
end

"""
This function solves for the economy with aggregate fluctuations as a fixed point for the law of motion of aggregate capital

##### Arguments
* `m::KrusellSmith`: Object holding the parameter values of the Krusell Smith economy
* `a::Approx`: Object holding the approximation elements used in the solution algorithm
* `ss::SteadySolution`: Object holding the steady-state solution elements
* `;maxiter::Int64(30)`: Maximum number of iterations to find fixed point
* `;tol::Float64(1e-3)`: Tolerance to stop fixed point iteration
* `;Tsim::Int64(11000)`: Number of simulation periods of the economy
* `;burn::Int64(1000)`: Number of simulation periods dropped to purge the effect of initial conditions
* `;w_lm::Float64(0.5)`: Weight on current law of motion parameters when updating to new law of motion
* `;seed::Int64(1)`: Seed for the random number generator used when simulating the exogenous aggregate state
* `;verbose::Bool(true)`: true = print text; false = do not print text

##### Returns
* `x::AggregateSolution`: Approximate solution to the economy with aggregate fluctuations
"""
function solve_fluctuations(m::KrusellSmith, a::Approx, ss::SteadySolution; maxiter = 30, tol = 1e-3, Tsim = 11000, burn = 1000, w_lm = 0.5, seed = 1, verbose = true)
    Pz, Pϵ = m.Pz, m.Pϵ
    nk, nϵ, nK, nz, gridk, cartkf_vec, gridK = a.nk, a.nϵ, a.nK, a.nz, a.gridk, vec(a.cartkf), a.gridK

    # given a history of K and a vector which selects periods when the exogenous aggregate state is s, get law-of-motion coefficients with OLS
    function get_lm_ols(Ksim::Vector{Float64}, sel_s::Vector{Int64})
        K_s = Ksim[sel_s]
        Kprime_s = Ksim[sel_s + 1]
        X = [ones(length(K_s), 1) log(K_s)]
        ((X' * X) \ X') * log(Kprime_s)
    end

    # initial guess for law of motion
    lm = ([log(ss.K); 0.0], [log(ss.K); 0.0])

    # simulate history of exogenous aggregate state using QuantEcon function, then remove burn-in period and create selector vectors for bad and good states
    srand(seed)
    zsim = simulate(MarkovChain(Pz), Tsim, 2)
    zsim_burn = zsim[(burn + 1):(end - 1)]
    sel_b, sel_g = find(zsim_burn .== 1), find(zsim_burn .== 2)

    # initialize density at steady state
    μ = vec(ss.μ)

    polc = Array(Float64, nk, nϵ, nK, nz)
    polk = similar(polc)

    for n in 1:maxiter
        Ksim = similar(zsim, Float64)
        # initialize aggregate capital at steady state
        Ksim[1] = ss.K

        # solve the household's problem, capping extrapolation at kmax to have well-defined transition
        polk, polc = solve_policies(m, a, lm)
        itp_polk = interpolate((gridk, 1:nϵ, gridK, 1:nz), polk, (Gridded(Linear()), NoInterp(), Gridded(Linear()), NoInterp()))

        for t in 1:(Tsim - 1)
            iz, izprime = zsim[t:(t + 1)]
            K = Ksim[t]
            # pull out transition matrix of exogenous idiosyncratic state depending on s and s'
            Pϵ_ssprime = (Pϵ[iz])[izprime]

            # update density on idiosyncratic states
            μ = update_density(m, a, μ, K, iz, itp_polk, Pϵ_ssprime)
            # compute aggregate capital
            Ksim[t + 1]  = dot(cartkf_vec, μ)
        end
        # remove burn-in period from aggregate capital simulation
        Ksim = Ksim[(burn + 1):end]

        # get new law-of-motion coefficients using OLS
        lm_ols = (get_lm_ols(Ksim, sel_b), get_lm_ols(Ksim, sel_g))

        # update law-of-motion coefficients as weighted average of current and new coefficients
        lm_next_b, lm_next_g = w_lm * lm[1] + (1 - w_lm) * lm_ols[1], w_lm * lm[2] + (1 - w_lm) * lm_ols[2]

        err = maxabs([lm_next_b - lm[1]; lm_next_g - lm[2]])
        lm = (lm_next_b, lm_next_g)

        if verbose
            @printf("Iter. # %d, Dist. = %1.2e; Capital in simul., Min: %1.2f, Max: %1.2f\n", n, err, minimum(Ksim), maximum(Ksim))
            @printf(" -- α_b = %1.4f, α_g = %1.4f, β_b = %1.4f, β_g = %1.4f\n", (lm[1])[1], (lm[2])[1], (lm[1])[2], (lm[2])[2])
            @printf("\n")
        end
        if n == maxiter; warn("Aggregate law of motion solution: hit maxiter"); end

        if err < tol; break; end
    end

    @printf("Economy with aggregate fluctuations done!\n")
    AggregateSolution(lm, polk, polc)
end

"""
This function simulates the economy with aggregate fluctuations

##### Arguments
* `m::KrusellSmith`: Object holding the parameter values of the Krusell Smith economy
* `a::Approx`: Object holding the approximation elements used in the solution algorithm
* `ss::SteadySolution`: Object holding the steady-state solution elements
* `fl::AggregateSolution`: Object holding the solution elements for the economy with aggregate fluctuations
* `;Tsim::Int64(6000)`: Number of simulation periods of the economy
* `;burn::Int64(1000)`: Number of simulation periods dropped to purge the effect of initial conditions
* `;seed::Int64(10)`: Seed for the random number generator used when simulating the exogenous aggregate state

##### Returns
* `zsim_burn::Vector{Int64}`: Simulated path of exogenous aggregate productivity (state numbers)
* `Ksim::Vector{Float64}`: Simulated path of aggregate capital
* `μ::Matrix{Float64}`: Density on idiosyncratic states in the last simulation period
"""
function simulate_fluctuations(m::KrusellSmith, a::Approx, ss::SteadySolution, fl::AggregateSolution; Tsim = 6000, burn = 1000, seed = 10)
    Pz, Pϵ = m.Pz, m.Pϵ
    nϵ, nz, gridk, cartkf_vec, gridK = a.nϵ, a.nz, a.gridk, vec(a.cartkf), a.gridK
    polk = fl.polk

    srand(seed)
    zsim = simulate(MarkovChain(Pz), Tsim, 2)
    zsim_burn = zsim[(burn + 1):(end - 1)]
    Ksim = similar(zsim, Float64)

    # initialize economy at steady state
    Ksim[1] = ss.K
    μ = vec(ss.μ)

    itp_polk = interpolate((gridk, 1:nϵ, gridK, 1:nz), polk, (Gridded(Linear()), NoInterp(), Gridded(Linear()), NoInterp()))
    for t in 1:(Tsim - 1)
        iz, izprime = zsim[t:(t + 1)]
        K = Ksim[t]
        # pull out transition matrix of exogenous idiosyncratic state depending on s and s'
        Pϵ_ssprime = (Pϵ[iz])[izprime]

        # update density on idiosyncratic states
        μ = update_density(m, a, μ, K, iz, itp_polk, Pϵ_ssprime)
        # compute aggregate capital
        Ksim[t + 1]  = dot(cartkf_vec, μ)
    end
    # remove burn-in period from aggregate capital simulation
    Ksim = Ksim[(burn + 1):end]

    zsim_burn, Ksim, μ
end

# ----------------- #
# Accuracy measures #
# ----------------- #

"""
This function computes Euler equation errors over the entire state space for the approximate policies in the steady state

##### Arguments
* `m::KrusellSmith`: Object holding the parameter values of the Krusell Smith economy
* `a::Approx`: Object holding the approximation elements used in the solution algorithm
* `ss::SteadySolution`: Object holding the steady-state solution elements
* `;nk_Eerr::Int64(1000)`: Number of evenly-spaced values of capital holding where Euler equation errors are computed

##### Returns
* `gridk_Eerr::Vector{Float64}`: Capital-holding values where Euler error is computed
* `x::Matrix{Float64}`: Euler equation error in log10 units
"""
function Euler_errors(m::KrusellSmith, a::Approx, ss::SteadySolution; nk_Eerr = 1000)
    β, σ, Pϵss = m.β, m.σ, m.Pϵss
    kmin, kmax, gridk, nϵ = a.kmin, a.kmax, a.gridk, a.nϵ
    R, polk, polc = (1 + ss.r), ss.polk, ss.polc

    itp_polc = interpolate((gridk, 1:nϵ), polc, (Gridded(Linear()), NoInterp()))
    itp_polk = extrapolate(interpolate((gridk, 1:nϵ), polk, (Gridded(Linear()), NoInterp())), Interpolations.Throw())

    gridk_Eerr = collect(linspace(10 * kmin, kmax, nk_Eerr))
    Eerr = Array(Float64, nk_Eerr, nϵ)
    for iϵ in 1:nϵ, ik in 1:nk_Eerr
        # pull out capital-holding value where Euler error is computed
        k = gridk_Eerr[ik]
        # interpolate capital-holding policy on that state value
        kprime = itp_polk[k, iϵ]
        # get Euler equation error
        Eerr[ik, iϵ] = 1 - (β * R * dot([itp_polc[kprime, 1]; itp_polc[kprime, 2]].^(- σ), vec(Pϵss[iϵ, :])))^(- 1 / σ) / itp_polc[k, iϵ]
    end

    gridk_Eerr, log10(abs(Eerr))
end

"""
This function computes Euler equation errors over the entire state space for the approximate policies in the economy with aggregate fluctuations

##### Arguments
* `m::KrusellSmith`: Object holding the parameter values of the Krusell Smith economy
* `a::Approx`: Object holding the approximation elements used in the solution algorithm
* `ss::SteadySolution`: Object holding the steady-state solution elements
* `fl::AggregateSolution`: Object holding the solution elements for the economy with aggregate fluctuations
* `;ndraws::Int64(100)`: Number of draws from the stationary distribution of aggregate states
* `;Tsim::Int64(1500)`: Number of simulation periods to generate draws from the stationary distribution of aggregate states
* `;nk_Eerr::Int64(1000)`: Number of evenly-spaced values of capital holding where Euler equation errors are computed

##### Returns
* `gridk_Eerr::Vector{Float64}`: Capital-holding values where Euler error is computed
* `x::Matrix{Float64}`: Average Euler equation error in log10 units
* `y::Matrix{Float64}`: Maximum Euler equation error in log10 units
"""
function Euler_errors(m::KrusellSmith, a::Approx, ss::SteadySolution, fl::AggregateSolution; ndraws = 100, Tsim = 1500, nk_Eerr = 1000)
    β, σ, Pz, Pϵ = m.β, m.σ, m.Pz, m.Pϵ
    kmin, kmax, gridk, nϵ, gridK, nz, gridz, gridL = a.kmin, a.kmax, a.gridk, a.nϵ, a.gridK, a.nz, a.gridz, a.gridL
    lm, polk, polc = fl.lm, fl.polk, fl.polc

    itp_polc = interpolate((gridk, 1:nϵ, gridK, 1:nz), polc, (Gridded(Linear()), NoInterp(), Gridded(Linear()), NoInterp()))
    itp_polk = extrapolate(interpolate((gridk, 1:nϵ, gridK, 1:nz), polk, (Gridded(Linear()), NoInterp(), Gridded(Linear()), NoInterp())), Interpolations.Throw())

    gridk_Eerr = collect(linspace(10 * kmin, kmax, nk_Eerr))

    # focus on average and maximum Euler equation error across draws of aggregate states at each pre-determined value of capital holding
    av_Eerr, max_Eerr = zeros(nk_Eerr, nϵ), zeros(nk_Eerr, nϵ)

    for n in 1:ndraws
        if (n % 20 == 0); @printf("%d draws done\n", n); end
        # generate draw from the stationary distribution of aggregate states (seed changes for each draw)
        zsim, Ksim, xx = simulate_fluctuations(m, a, ss, fl; Tsim = Tsim, seed = 1 + n)

        for iϵ in 1:nϵ, ik in 1:nk_Eerr
            margU = 0.0
            # pull out capital-holding value where Euler error is computed, together with the draw of aggregate states
            k, K, Kprime, iz = gridk_Eerr[ik], Ksim[end-1], Ksim[end], zsim[end-1]
            # interpolate capital-holding policy on that state value
            kprime = itp_polk[k, iϵ, K, iz]
            # get Euler equation error
            for izprime in 1:nz
                zprime, Lprime = gridz[izprime], gridL[izprime]
                Rprime = irate(m, zprime, Kprime, Lprime) + 1
                margU += Pz[iz, izprime] * β * Rprime * dot([itp_polc[kprime, 1, Kprime, izprime]; itp_polc[kprime, 2, Kprime, izprime]].^(- σ), vec(((Pϵ[iz])[izprime])[iϵ, :]))
            end
            Eerr = abs(1 - margU^(- 1 / σ) / itp_polc[k, iϵ, K, iz])
            av_Eerr[ik, iϵ] += Eerr
            max_Eerr[ik, iϵ] = max(Eerr, max_Eerr[ik, iϵ])
        end
    end

    gridk_Eerr, log10(av_Eerr / ndraws), log10(max_Eerr)
end

"""
This function computes several accuracy measures for the law of motion of the economy with aggregate fluctuations

##### Arguments
* `m::KrusellSmith`: Object holding the parameter values of the Krusell Smith economy
* `a::Approx`: Object holding the approximation elements used in the solution algorithm
* `ss::SteadySolution`: Object holding the steady-state solution elements
* `fl::AggregateSolution`: Object holding the solution elements for the economy with aggregate fluctuations
* `;Tsim::Int64(11000)`: Number of simulation periods of the economy
* `;burn::Int64(1000)`: Number of simulation periods dropped to purge the effect of initial conditions
"""
function lm_errors(m::KrusellSmith, a::Approx, ss::SteadySolution, fl::AggregateSolution; Tsim = 11000, burn = 1000)
    lm = fl.lm

    # simulate a path for the economy (seed is set by keyword argument to be different than in the fixed point iterative solution; no burn-in period)
    zsim, Ksim, xx = simulate_fluctuations(m, a, ss, fl; Tsim = Tsim, burn = 0)
    lKsim = log(Ksim)

    lKlm, lK1step = similar(Ksim), similar(Ksim)
    lKlm[1] = lKsim[1]
    lK1step[1] = lKsim[1]
    for t in 1:(length(Ksim) - 1)
        iz = zsim[t]
        # get path for aggregate capital by using the law of motion only
        lKlm[t + 1] = dot(lm[iz], [1.0; lKlm[t]])
        # get sequence of 1-period forecasts from the law of motion
        lK1step[t + 1] = dot(lm[iz], [1.0; lKsim[t]])
    end

    # remove burn-in period
    zsim = zsim[(burn + 1):end]
    lKsim = lKsim[(burn + 1):end]
    lKlm = lKlm[(burn + 1):end]
    lK1step = lK1step[(burn + 1):end]

    ldiff = abs(lKsim - lKlm)
    resid = lKsim - lK1step

    # compute log-difference statistics based on the law of motion only
    mean_ldiff = mean(ldiff)
    max_ldiff = maximum(ldiff)

    # compute R-squared and RMSE of 1-period forecasts, separately for bad and good states
    sel_b, sel_g = find(zsim[1:(end - 1)] .== 1) + 1, find(zsim[1:(end - 1)] .== 2) + 1
    R2_b = 1 - sum(resid[sel_b].^2) / sum((lKsim[sel_b] - mean(lKsim[sel_b])).^2)
    R2_g = 1 - sum(resid[sel_g].^2) / sum((lKsim[sel_g] - mean(lKsim[sel_g])).^2)
    RMSE_b = sqrt(mean(resid[sel_b].^2))
    RMSE_g = sqrt(mean(resid[sel_g].^2))

    @printf("Mean log difference = %1.4f, Maximum log difference = %1.4f\n", mean_ldiff, max_ldiff)
    @printf("Bad state: R-squared = %1.6f, RMSE = %1.6f\n", R2_b, RMSE_b)
    @printf("Good state: R-squared = %1.6f, RMSE = %1.6f\n", R2_g, RMSE_g)
end

# -----------#
# Statistics #
# -----------#

"""
This function computes the cross sectional statistics (wealth shares, Gini for wealth, Lorenz curve for wealth) used in Krusell Smith 1998

##### Arguments
* `m::KrusellSmith`: Object holding the parameter values of the Krusell Smith economy
* `a::Approx`: Object holding the approximation elements used in the solution algorithm
* `ss::SteadySolution`: Object holding the steady-state solution elements
* `fl::AggregateSolution`: Object holding the solution elements for the economy with aggregate fluctuations
* `;ndraws::Int64(100)`: Number of draws from the stationary distribution of aggregate states
* `;Tsim::Int64(1500)`: Number of simulation periods to generate draws from the stationary distribution of aggregate states

##### Returns
* `x::Vector{Float64}`: Percentiles of wealth distribution
* `y::Vector{Float64}`: Values of Lorenz curve for wealth
"""
function stats_crosssection(m::KrusellSmith, a::Approx, ss::SteadySolution, fl::AggregateSolution; ndraws = 100, Tsim = 1500)
    nkf, nϵ = a.nkf, a.nϵ
    # create arrays of percentiles to compute wealth shares (alone or in Gini for wealth)
    pl_list = [0.99; 0.95; 0.9; 0.8; 0.7]
    pu_list = [collect(0.1:0.1:0.9); 0.95; 0.99]

    # given capital-holding density, aggregate capital and percentile bounds, compute wealth share
    function wealth_share(a::Approx, μk::Vector{Float64}, K::Float64, pl::Float64, pu::Float64)
        gridkf = a.gridkf
        cum_μk = cumsum(μk)
        sel = (cum_μk .<= pu) .* (cum_μk .> pl)
        dot(μk[sel], gridkf[sel]) / K
    end

    # recursive formula to compute Gini for wealth using the capital-holding density
    function gini(a::Approx, μk::Vector{Float64})
        gridkf = a.gridkf
        term1 = cumsum(gridkf .* μk)
        term2 = term1[1] * μk[1] + sum(μk[2:end] .* (term1[2:end] + term1[1:(end - 1)]))
        return 1 - term2 / term1[end]
    end

    av_wealth_shares, av_gini, av_lorenz = zeros(length(pl_list)), 0.0, zeros(length(pu_list))
    for n in 1:ndraws
        if (n % 20 == 0); @printf("%d draws done\n", n); end
        # generate draw from the stationary distribution of aggregate states (seed changes for each draw)
        zsim, Ksim, μ = simulate_fluctuations(m, a, ss, fl; Tsim = Tsim, seed = 1 + n)
        K, μk = Ksim[end], vec(sum(reshape(μ, nkf, nϵ), 2))

        # add current draw of all statistics in order to average afterwards with respect to the aggregate states
        av_wealth_shares += convert(Array{Float64,1}, [wealth_share(a, μk, K, pl, 1.0) for pl in pl_list])
        av_gini += gini(a, μk)
        av_lorenz += convert(Array{Float64,1}, [wealth_share(a, μk, K, 0.0, pu) for pu in pu_list])
    end

    av_wealth_shares = av_wealth_shares / ndraws * 100
    av_gini = av_gini / ndraws
    av_lorenz = av_lorenz / ndraws

    @printf("\n")
    @printf("Wealth shares: top 1 = %1.1f, top 5 = %1.1f, top 10 = %1.1f, top 20 = %1.1f, top 30 = %1.1f\n", av_wealth_shares[1], av_wealth_shares[2], av_wealth_shares[3], av_wealth_shares[4], av_wealth_shares[5])
    @printf("\n")
    @printf("Gini coefficient for wealth = %1.3f\n", av_gini)

    [0.0; pu_list; 1.0], [0.0; av_lorenz; 1.0]
end

"""
This function computes the aggregate variable statistics used in Krusell Smith 1998

##### Arguments
* `m::KrusellSmith`: Object holding the parameter values of the Krusell Smith economy
* `a::Approx`: Object holding the approximation elements used in the solution algorithm
* `ss::SteadySolution`: Object holding the steady-state solution elements
* `fl::AggregateSolution`: Object holding the solution elements for the economy with aggregate fluctuations
* `;Tsim::Int64(11000)`: Number of simulation periods of the economy
"""
function stats_aggregate(m::KrusellSmith, a::Approx, ss::SteadySolution, fl::AggregateSolution; Tsim = 11000)
    α, δ = m.α, m.δ
    gridz, gridL = a.gridz, a.gridL

    # simulate a path for the economy (seed is set by keyword argument to be different than in the fixed point iterative solution; default burn-in period)
    zsim, Ksim, xx = simulate_fluctuations(m, a, ss, fl; Tsim = Tsim)

    Y = Array(Float64, (length(Ksim) - 1))
    C, I = similar(Y), similar(Y)
    for t in 1:length(Y)
        z, L = gridz[zsim[t]], gridL[zsim[t]]

        # compute the aggregate variables of interest: output, investment and consumption
        Y[t], I[t] = z * Ksim[t]^α * L^(1- α), Ksim[t+1] - (1 - δ) * Ksim[t]
        C[t] = Y[t] - I[t]
    end

    # compute the statistics of interest
    mean_K = mean(Ksim)
    corr_CY = crosscor(C, Y)[1]
    std_I = std(I)
    yearly_autocorr_Y = autocor(Y, [4])[1]

    @printf("Mean capital = %1.2f, Consumption-output correlation = %1.3f\n", mean_K, corr_CY)
    @printf("\n")
    @printf("Standard dev. investment = %1.3f, Yearly autocorrelation output = %1.3f\n", std_I, yearly_autocorr_Y)
end

# -------------------#
# Plotting functions #
# -------------------#

include("KrusellSmith_plots.jl")

end # module
