using DataFrames
using OrderedCollections: OrderedDict
using CSV
using SparseArrays

# global variables
BASE_POWER = 1000  # could also be be imported with infra, as nominal sys power
BASE_VOLTAGE = 230
H_MAX = 5
HARMONICS = [h for h in 1:2:H_MAX]
NET_FREQ = 50
HARMONICS_FREQ = [NET_FREQ * i for i in HARMONICS]
PATH = "harmonic-power-flow\\Harmonic Power Flow\\"

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
        nodes = import_nodes_from_csv(PATH*name)
        lines = import_lines_from_csv(PATH*name)
    end
    # find first nonlinear bus
    m = minimum(nodes[nodes[:,"type"] .== "nonlinear",:].ID)
    n = size(nodes, 1)
    nodes, lines, m, n
end


"""
    admittance_matrices(nodes, lines, harmonics)

Build the nodal admittance matrices (admittance laplacian) for all harmonics.
"""
function admittance_matrices(nodes, lines, harmonics)
    LY = Dict()
    for h in harmonics
        LY[h] = spzeros(Complex, n,n)
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


function init_fund_state_vec(u)
    xv = u[1].v
    xϕ = u[1].ϕ
    for h in HARMONICS[2:end]
        xv = vcat(xv, u[h].v)
        xϕ = vcat(xϕ, u[h].ϕ)
    end
    vcat(xv, xϕ)
end


function fund_mismatch(nodes, u, LY_1)
    u1 = u[1].v .* exp.(1im*u[1].ϕ)
    s = (nodes.P + 1im*nodes.Q)/BASE_POWER
    mismatch = u1 .* conj(LY_1*u1) + s
    f = vcat(real(mismatch[2:end]), imag(mismatch[2:end]))
    err = maximum(f)
    f, err
end

nodes, lines, m, n = init_network("net2")
LY = admittance_matrices(nodes, lines, HARMONICS)
u = init_voltages(nodes, HARMONICS)
x = init_fund_state_vec(u)
f, err_f = fund_mismatch(nodes, u, LY[1])