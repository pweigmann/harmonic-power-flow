using DataFrames
using OrderedCollections: OrderedDict
using CSV
using SparseArrays

"""
requires Julia 1.6 (spdiagm)

notation
u: complex voltage
v: voltage magnitude
ϕ: voltage phase

"""

# global variables
BASE_POWER = 1000  # could also be be imported with infra, as nominal sys power
BASE_VOLTAGE = 230
H_MAX = 5
HARMONICS = [h for h in 1:2:H_MAX]
NET_FREQ = 50
HARMONICS_FREQ = [NET_FREQ * i for i in HARMONICS]
NET_PATH = "harmonic-power-flow\\Harmonic Power Flow\\"

# pu system derived base values
base_current = 1000*BASE_POWER/BASE_VOLTAGE
base_admittance = base_current/BASE_VOLTAGE

# number of harmonics (without fundamental)
K = length(HARMONICS) - 1

function import_nodes_from_csv(filename)
    df = CSV.read(filename * "_buses.csv", DataFrame)
end


function import_lines_from_csv(filename)
    df = CSV.read(filename * "_lines.csv", DataFrame)
end


function import_nodes_manually()
    nodes = DataFrame(
        ID = 1:5, 
        type = ["slack", "PQ", "PQ", "PQ", "nonlinear"], 
        component = ["generator", "lin_load_1", "lin_load_2", nothing, "smps"],
        S = [1000, nothing, nothing, nothing, nothing],
        X_shunt = [0.0001, 0, 0, 0, 0],
        P1 = [nothing, 100, 100, 0, 250],
        Q1 = [nothing, 100, 100, 0, 100])
end


function import_lines_manually()
    lines = DataFrame(
        ID = 1:5,
        fromID = 1:5,
        toID = [2,3,4,5,1],
        R1 = [0.01, 0.02, 0.01, 0.01, 0.01],
        X1 = [0.01, 0.08, 0.02, 0.02, 0.02])
end


function init_network(name, manually=false)
    if manually
        nodes = import_nodes_manually()
        lines = import_lines_manually()
    else
        nodes = import_nodes_from_csv(NET_PATH*name)
        lines = import_lines_from_csv(NET_PATH*name)
    end
    # find first nonlinear bus
    m = minimum(nodes[nodes[:,"type"] .== "nonlinear",:].ID)
    n = size(nodes, 1)
    nodes, lines, m, n
end


"""
    admittance_matrices(nodes, lines, harmonics)

Build the nodal admittance matrices (admittance laplacian) for all harmonics. Admittance scales linearly with frequency.
"""
function admittance_matrices(nodes, lines, harmonics)
    LY = Dict()
    for h in harmonics
        LY[h] = spzeros(ComplexF64, n,n)
        # non-diagonal elements
        for line in eachrow(lines)
            LY[h][line.fromID, line.toID] = -1/(line.R + 1im*line.X*h)
            # nodal admittance matrix is assumed to be symmetric
            LY[h][line.toID, line.fromID] = -1/(line.R + 1im*line.X*h)
        end
        # diagonal elements
        for i in 1:n
            if nodes.X_shunt[i] > 0 && h != 1
                LY[h][i, i] = -sum(LY[h][i, :]) + 1/(1im*nodes.X_shunt[i]*h)
            else
                LY[h][i, i] = -sum(LY[h][i, :])
            end
        end
    end
    LY
end


function init_voltages(nodes, harmonics, v1=1, ϕ1 = 0, vh=0.1, ϕh=0)
    u = Dict()
    for h in harmonics
        if h == 1
            u[h] = DataFrame(
                v = ones(size(nodes, 1))*v1,
                ϕ = ones(size(nodes, 1))*ϕ1
            )  
        else
            u[h] = DataFrame(
                v = ones(size(nodes, 1))*vh,
                ϕ = ones(size(nodes, 1))*ϕh
            )
        end
    end
    u
end


function fund_state_vec(u)
    xϕ = u[1].ϕ[2:end]
    xv = u[1].v[2:end]
    vcat(xϕ, xv)  # note: phase first
end


function fund_mismatch(nodes, u, LY)
    LY_1 = LY[1]
    u_1 = u[1].v .* exp.(1im*u[1].ϕ)
    s = (nodes.P + 1im*nodes.Q)/BASE_POWER
    mismatch = u_1 .* conj(LY_1*u_1) + s
    f = vcat(real(mismatch[2:end]), imag(mismatch[2:end]))
    err = maximum(f)
    f, err
end


function fund_jacobian(u, LY)
    LY_1 = LY[1]
    u_1 = u[1].v .* exp.(1im*u[1].ϕ)
    i_diag = spdiagm(sparse(LY_1 * u_1))
    u_1_diag = spdiagm(u_1)
    u_1_diag_norm = spdiagm(u_1./u_1)

    dSdϕ = 1im*u_1_diag*(i_diag - LY_1 * u_1_diag)'
    dSdv = u_1_diag_norm * i_diag' + u_1_diag * (LY_1 * u_1_diag_norm)'

    # divide sub-matrices into real and imag part, cut off slack
    dPdϕ = real(dSdϕ[2:end, 2:end])
    dPdv = real(dSdv[2:end, 2:end])
    dQdϕ = imag(dSdϕ[2:end, 2:end])
    dQdv = imag(dSdv[2:end, 2:end])

    J_1 = vcat(hcat(dPdϕ, dPdv), 
               hcat(dQdϕ, dQdv))
end


function update_fund_state_vec(J, x, f)
    x_new = x - J\f  # Newton-Raphson iteration
end


function update_fund_voltages(u, x)
    u[1].ϕ[2:end] = x[1:(length(x)÷2)]
    u[1].v[2:end] = x[(length(x)÷2+1):end]
    u
end


function pf(LY, nodes, thresh_f = 1e-6, max_iter_f = 30, 
            plt_convergence = false)
    u = init_voltages(nodes, HARMONICS)
    x_f = fund_state_vec(u)
    f_f, err_f = fund_mismatch(nodes, u, LY)

    n_iter_f = 0
    err_f_t = Dict()
    while err_f > thresh_f && n_iter_f <= max_iter_f
        J_f = fund_jacobian(u, LY)
        x_f = update_fund_state_vec(J_f, x_f, f_f)
        u = update_fund_voltages(u, x_f)
        f_f, err_f = fund_mismatch(nodes, u, LY)
        err_f_t[n_iter_f] = err_f
        n_iter_f += 1
    end

    if n_iter_f < max_iter_f
        println("Fundamental power flow converged after ", n_iter_f, 
              " iterations.")
    elseif n_iter_f == max_iter_f
        println("Maximum of ", n_iter_f, " iterations reached.")
    end
    u, err_f_t, n_iter_f
end


# Harmonic Power Flow functions
function import_Norton_Equivalents(nodes, coupled=true, folder_path="")
    NE = Dict()
    nl_components = unique(nodes[nodes.type .== "nonlinear", "component"])
    for device in nl_components
        NE_df = CSV.read("harmonic-power-flow\\Circuit Simulation\\" * device * "_NE.csv", DataFrame)
        # transform to Complex type, enough to strip first paranthesis for successful parse
        values = mapcols!(col -> parse.(ComplexF64, strip.(col, ['('])), NE_df[:, 3:end])
        NE_device = hcat(NE_df[:,1:2], values)

        if coupled
            I_N = Array(NE_device[NE_device.Parameter .== "I_N_c",3:end])/base_current
            Y_N = Array(NE_device[NE_device.Parameter .== "Y_N_c",3:end])/base_admittance
        else
            I_N = Array(NE_device[NE_device.Parameter .== "I_N_uc",3:end])/base_current
            Y_N = Array(NE_device[NE_device.Parameter .== "Y_N_uc",3:end])/base_admittance
        end    
        NE[device] = [I_N, Y_N]
    end
    NE
end

nodes, lines, m, n = init_network("net2")
LY = admittance_matrices(nodes, lines, HARMONICS)
u, err_f_t, n_iter_f = pf(LY, nodes)
println(u[1])
NE = import_Norton_Equivalents(nodes)


# function harmonic_state_vec(u)
#     xv = u[1].v[2:end]
#     xϕ = u[1].ϕ[2:end]
#     for h in HARMONICS[2:end]
#         xv = vcat(xv, u[h].v)
#         xϕ = vcat(xϕ, u[h].ϕ)
#     end
#     vcat(xv, xϕ)  # note: magnitude first
# end