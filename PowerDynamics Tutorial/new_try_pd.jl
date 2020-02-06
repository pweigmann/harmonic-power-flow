# %%
# run this only once in a session
push!(LOAD_PATH, "./PowerDynamics Tutorial\\");
LOAD_PATH
using Functions
# %%
using SparseArrays
using LinearAlgebra
using Statistics
using Random
using LaTeXStrings
using Plots
# %%
Random.seed!(42); #random number generator seed
# %%
using PowerDynamics
using PowerDynOperationPoint
# %%
function pi_model(y, y_sh, t_uv, t_vu)
    # admittance and shunt are assumed to be symmetric
    B = spzeros(Complex{Float64}, 2, 2)
    B[1, 1] = abs2(t_uv) * (y + y_sh)
    B[1, 2] = - conj(t_uv) * t_vu * y
    B[2, 1] = - conj(t_vu) * t_uv * y
    B[2, 2] = abs2(t_vu) * (y + y_sh)
    return B
end
# %%
# grid parameters
N = 3
M = 2

# nominal values
s_nom = 800e6 #in W global
f_nom = 50 # Hz global

# low-voltage side
v_nom_l = 15e3 # in V
@show y_nom_l = s_nom / v_nom_l^2

# high-voltage side
v_nom_h = 400e3 # in V
@show y_nom_h = s_nom / v_nom_h^2

# the transformer relates the base voltages
@show t = v_nom_h / v_nom_l + 0im
# %% markdown
# Transformator:
# - Oberspannung: 400 kV
# - Unterspannung: 15 kV
# - Nennleistung: 800 MVA
# - Schaltgruppe OS-Seite: YN
# - Schaltgruppe US-Seite: D
#
# - X1 = 0.15 p.u. (bezogen auf OS-Seite) oder X1 = 30 Ohm (bezogen auf OS-Seite)
# - Bezugsimpedanz: (400kV)²/800MVA = 200 Ohm
# - R1 = 0 Ohm
# - X1 = 0.15 p.u. oder Kurzschlussspannung uk = 15 % mit X/R-Verhältnis gegen unendlich
# %%
# transformer
@assert y_nom_h * 30 == 0.15
@show yt = yrx(0, 30, pu=y_nom_h)
mixed_pu = [[1. t];[t t^2]] # the trafo connects nodes with different reference
Bt = pi_model(yt, 0, 1, t)  .* mixed_pu
# %% markdown
# Leitung:
# - Nennspannung: 400kV
# - Nennstrom: 2,02 kA
# - R: 8 Ohm
# - X: 80 Ohm
# - B: 936,2 µS
# - Leitungslänge: 1 km
# %%
# transmission line
@show yl = yrx(8, 80, pu=y_nom_h) # input in Ω
@show yl_sh = ygb(0, 936.2e-6, pu=y_nom_h) # input in S
Bl = pi_model(yl, yl_sh, 1, 1)
# %%
Y = spzeros(Complex, N, N)
Y[1:2, 1:2] -= Bt
Y[2:3, 2:3] -= Bl
Y = Y * spdiagm(0 => -1. .* ones(Complex, N))
# %%
passive = [false, true, false] # should be somehow automated
Yred = kron_reduction(Y, passive)
# %% markdown
# Synchronmaschine:
# - Nennscheinleistung: 800 MVA
# - Nennspannung: 15kV
# - cos(phi) = 0.95
# - Arbeitspunkt: 600 MW Einspeisung mit Spannungsregelung  an US-Seite des Trafos (1 p.u.)
# - weitere Daten auf der nächsten Seite!
# %%
SyncM = FourthEq(
                    H = 3,
                    P = 600e6 / s_nom,
                    D = 0.1,
                    T_d_dash = 5,
                    T_q_dash = 0.1,
                    X_d = 1.1,
                    X_q = 0.7,
                    X_d_dash =0.25,
                    X_q_dash =0.7,
                    Ω=50.,
                    E_f=1. # should be 1pu on the low-voltage side?
)
# %% markdown
# Externes Netz (wie Synchronmaschine):
# - Anlaufzeitkonstante: 999999s
# - Subtransiente Kurzschlussleistung: 10000 MVA
# - R/X- Verhältnis: 0,1
# %%
ExternalGrid = SlackAlgebraic(U=1)
# %%
# create network dynamics object
g = GridDynamics([SyncM, ExternalGrid], Yred, skip_LY_check=true)
# %%
# find the fixed point = normal operation point
guess = State(g, 0.1randn(SystemSize(g)));
fp = getOperationPoint(g, guess)
# %%
x0 = copy(guess);
x0[1, :u] = exp(deg2rad(23.8) * 1im) * 15e3 / v_nom_l
x0[2, :u] = 1. + 0im
# %%
using DifferentialEquations
# %%
timespan = (0.0, 10.)
sol = solve(g, x0, timespan);
# %%
pl_v = plot(sol, :, :v, legend = (0.8, 1.), ylabel=L"V [p.u.]")
pl_p = plot(sol, :, :φ, legend = (0.8, 0.95), ylabel=L"\phi [rad]")
plot!(sol, 1, :θ, legend = (0.8, 0.95), label=L"\theta [rad] (rotor angle)")
pl_ω = plot(sol, 1, :ω, legend = (0.8, 0.7), ylabel=L"\omega \left[rad/s\right]")
pl = plot(
    pl_v, pl_p, pl_ω;
    layout=(3,1),
    size = (500, 500),
    lw=3,
    xlabel=L"t[s]"
)
# %%
sol.dqsol[:, end]
# %% markdown
# __Fehler__:
# - Dreiphasiger Kurzschluss an Sammelschiene 4012 (bei Sekunde 1), Dauer: 0,189 s,
# - Fehlerwiderstand: R = 0,001 Ohm und X = 0,1 Ohm
# %%
# line fault admittance -> additional shunt at node 4012
@show yfault = yrx(0.001, 0.1, pu=y_nom_h); # in Ω
# %%
Y_fault = copy(Y)
Y_fault[3, 3] += yfault
# %%
passive = [false, true, false] # should be somehow automated
Yred_fault = 1im .* kron_reduction(Y_fault, passive)
# %%
Yred_fault
# %%
import Base: @__doc__
import PowerDynBase.construct_node_dynamics, PowerDynBase.showdefinition, PowerDynBase.ODENodeSymbols
# %%
@DynamicNode Shunts(Y_sh) <: OrdinaryNodeDynamicsWithMass(m_u=false, m_int=no_internal_masses)  begin
end [] begin
    du =  - i ./ conj(Y_sh)
end
# %%
Fault = Shunts(Y_sh=yfault)
# %%
g_fault = GridDynamics([SyncM, SlackAlgebraic(U=0)], Yred_fault, skip_LY_check=true)
# %%
x0 = operationpoint(g_fault, randn(rng, SystemSize(g_fault))) # just to create the right type for x0
#for i in 1:SystemSize(g_fault)
#    x0.base.vec[i] = sol.dqsol[i, end]
#end
# %%
x0[1, :u] = exp(deg2rad(23.8) * 1im) * 15e3 / v_nom_l
x0[2, :u] = 1. + 0im
x0[1, :int, 2] = 0
# %%
timespan = (0.0, .1)
sol = solve(g_fault, x0, timespan);
# %%
pl_v = plot(sol, 1, :v, legend = (0.8, 1.), ylabel=L"V [p.u.]")
pl_p = plot(sol, 1, :φ, legend = (0.8, 0.95), ylabel=L"\phi [rad]")
pl_ω = plot(sol, 1, :ω, legend = (0.8, 0.7), ylabel=L"\omega \left[rad/s\right]")
pl_θ = plot(sol, 1, :θ, legend = (0.8, 0.7), ylabel=L"\theta \left[rad\right]")
pl = plot(
    pl_v, pl_p, pl_ω, pl_θ;
    layout=(2,2),
    size = (700, 500),
    lw=3,
    xlabel=L"t[s]"
)
# %%
sol(timespan[2], 1:1, :ω)
# %%
x0 = operationpoint(g, randn(rng, SystemSize(g))) # just to create the right type for x0
for i in 1:SystemSize(g)
    x0.base.vec[i] = sol.dqsol[i, end]
end
# %%
timespan = (0.0, 100.)
sol = solve(g, x0, timespan);
# %%
pl_v = plot(sol, :, :v, legend = (0.8, 1.), ylabel=L"V [p.u.]")
pl_p = plot(sol, :, :φ, legend = (0.8, 0.95), ylabel=L"\phi [rad]")
pl_ω = plot(sol, 1, :ω, legend = (0.8, 0.7), ylabel=L"\omega \left[rad/s\right]")
pl = plot(
    pl_v, pl_p, pl_ω;
    layout=(3,1),
    size = (500, 500),
    lw=3,
    xlabel=L"t[s]"
)
# %%
