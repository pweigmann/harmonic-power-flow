using DataFrames
using OrderedCollections: OrderedDict

# global variables
BASE_POWER = 1000  # could also be be imported with infra, as nominal sys power
BASE_VOLTAGE = 230
HARMONICS = [1, 5, 7]
NET_FREQ = 50
HARMONICS_FREQ = [NET_FREQ * i for i in HARMONICS]
MAX_ITER_F = 30  # maybe better as argument of pf function
MAX_ITER_H = 30
THRESH_F = 1e-6  # error threshold of fundamental mismatch function
THRESH_H = 1e-4
COUPLED_NE = true  # use Norton parameters of coupled vs. uncoupled model

# pu system
base_current = 1000*BASE_POWER/BASE_VOLTAGE
base_admittance = base_current/BASE_VOLTAGE

# number of harmonics (without fundamental)
K = length(HARMONICS) - 1

# buses = OrderedDict()
buses = DataFrame(
    ID = 1:5, 
    type = ["slack", "PQ", "PQ", "PQ", "nonlinear"], 
    component = ["generator", "lin_load_1", "lin_load_2", nothing, "smps"],
    S = [1000, nothing, nothing, nothing, nothing],
    X_shunt = [0.0001, 0, 0, 0, 0],
    P1 = [nothing, 100, 100, 0, 250],
    Q1 = [nothing, 100, 100, 0, 100])


lines = DataFrame(
    ID = 1:5,
    fromID = 1:5,
    toID = [2,3,4,5,1],
    R1 = [0.01, 0.02, 0.01, 0.01, 0.01],
    X1 = [0.01, 0.08, 0.02, 0.02, 0.02]
    )

# find first nonlinear bus (ID starts at "1")
m = minimum(buses[buses[:,"type"] .== "nonlinear",:].ID)
n = size(buses, 1)

function build_admittance_matrices(buses, lines, harmonics)
    Y = Dict()
    for h in harmonics
        Y[h] = DataFrame(zeros(Complex, n,n))  # column names not ideal
        # non diagonal matrix elements
        for line in eachrow(lines)
            Y[h][line.fromID, line.toID] = -1/(line.R1 + 1im*line.X1*h)
            # nodal admittance matrix is assumed to be symmetric
            Y[h][line.toID, line.fromID] = -1/(line.R1 + 1im*line.X1*h)
        end
        # diagonal elements
        for i in 1:n
            if buses.X_shunt[i] > 0 && h != 1
                Y[h][i, i] = -sum(Y[h][i, :]) + 1/(1im*buses.X_shunt[i]*h)
            else
                Y[h][i, i] = -sum(Y[h][i, :])
            end
        end
    end
    Y
end

Y_h = build_admittance_matrices(buses, lines, HARMONICS) 