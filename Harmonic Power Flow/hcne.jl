using CSV: length
using DataFrames
using OrderedCollections: OrderedDict
using CSV
using SparseArrays
using BlockArrays

"""
requires Julia 1.6 (spdiagm)

notation
u: complex voltage
v: voltage magnitude
ϕ: voltage phase

S1V1: Jacobian of derivatives of S at harmonic 1 wrt voltage magnitude at harmonic 1

n buses total (i = 1, ..., n)
slack bus is first bus (i = 1)
m-1 linear buses (i = 1, ..., m-1)
n-m+1 nonlinear buses (i = m, ..., n)
K harmonics considered (excluding fundamental)
L is last harmonic considered


TODO 
- harmonize usage of _1 or _f to reference fundamental frequency
- add docstrings, rework comments
"""

# global variables
NET_PATH = "harmonic-power-flow\\Harmonic Power Flow\\"


struct PowerGrid
    nodes
    lines
    m
    n
end


struct State
    u
    i
end


struct Settings
    coupled::Bool  # choose coupled or uncoupled Norton Equivalent model
    harmonics::Array  # only uneven harmonics, must contain "1", example: [1,3,5,7]
    base_frequency::Int  # in Hz, default: 50
    base_voltage::Number  # in V, default: 230
    base_power::Number  # in kVA, default: 1000

    # algorithm parameters
    thresh_f
    thresh_h
    max_iter_f::Int
    max_iter_h::Int

    # voltage start values [pu]
    v1 
    ϕ1 
    vh 
    ϕh

    # calculated parameters
    K::Int  # number of harmonics (without fundamental)
    base_current::Number
    base_admittance::Number
end


function init_settings(coupled, harmonics; base_frequency=50, base_voltage=230,
    base_power=1000, thresh_f = 1e-6, max_iter_f = 30, thresh_h=1e-4, max_iter_h=50,
    v1=1, ϕ1 = 0, vh=0.1, ϕh=0)

    K = length(harmonics) - 1

    # derived pu base values
    base_current = 1000*base_power/base_voltage
    base_admittance = base_current/base_voltage
    
    Settings(
        coupled, harmonics, base_frequency, base_voltage, base_power, 
        thresh_f, thresh_h, max_iter_f, max_iter_h, 
        v1, ϕ1, vh, ϕh, 
        K, base_current, base_admittance)
end


function init_power_grid(nodes::DataFrame, lines::DataFrame)
    m = minimum(nodes[nodes[:,"type"] .== "nonlinear",:].ID)  # number of linear nodes
    n = size(nodes, 1)  # total number of nodes

    PowerGrid(nodes, lines, m, n)
end


function import_nodes_from_csv(filename)
    CSV.read(filename * "_buses.csv", DataFrame)
end


function import_lines_from_csv(filename)
    CSV.read(filename * "_lines.csv", DataFrame)
end


function create_nodes_manually()
    DataFrame(
        ID = 1:5, 
        type = ["slack", "PQ", "PQ", "PQ", "nonlinear"], 
        component = ["generator", "lin_load_1", "lin_load_2", nothing, "smps"],
        S = [1000, nothing, nothing, nothing, nothing],
        X_shunt = [0.0001, 0, 0, 0, 0],
        P1 = [nothing, 100, 100, 0, 250],
        Q1 = [nothing, 100, 100, 0, 100])
end


function create_lines_manually()
    DataFrame(
        ID = 1:5,
        fromID = 1:5,
        toID = [2,3,4,5,1],
        R1 = [0.01, 0.02, 0.01, 0.01, 0.01],
        X1 = [0.01, 0.08, 0.02, 0.02, 0.02])
end


# function init_network(name, manually=false)
#     if manually
#         nodes = import_nodes_manually()
#         lines = import_lines_manually()
#     else
#         nodes = import_nodes_from_csv(NET_PATH*name)
#         lines = import_lines_from_csv(NET_PATH*name)
#     end
#     # find first nonlinear bus
#     m = minimum(nodes[nodes[:,"type"] .== "nonlinear",:].ID)
#     n = size(nodes, 1)
#     nodes, lines, m, n
# end


"""
    admittance_matrices(nodes, lines, harmonics)

Build the nodal admittance matrices (admittance laplacian) for all harmonics. Admittance scales linearly with frequency.
"""
function admittance_matrices(net, harmonics)
    LY = Dict()
    for h in harmonics
        LY[h] = spzeros(ComplexF64, net.n, net.n)
        # non-diagonal elements
        for line in eachrow(net.lines)
            LY[h][line.fromID, line.toID] = -1/(line.R + 1im*line.X*h)
            # nodal admittance matrix is assumed to be symmetric
            LY[h][line.toID, line.fromID] = -1/(line.R + 1im*line.X*h)
        end
        # diagonal elements
        for i in 1:net.n
            if net.nodes.X_shunt[i] > 0 && h != 1
                LY[h][i, i] = -sum(LY[h][i, :]) + 1/(1im*net.nodes.X_shunt[i]*h)
            else
                LY[h][i, i] = -sum(LY[h][i, :])
            end
        end
    end
    return LY
end


function init_voltages(nodes, settings)
    u = Dict()
    for h in settings.harmonics
        if h == 1
            u[h] = DataFrame(
                v = ones(size(nodes, 1))*settings.v1,
                ϕ = ones(size(nodes, 1))*settings.ϕ1
            )  
        else
            u[h] = DataFrame(
                v = ones(size(nodes, 1))*settings.vh,
                ϕ = ones(size(nodes, 1))*settings.ϕh
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


function fund_mismatch(nodes, u, LY, base_power)
    LY_1 = LY[1]
    u_1 = u[1].v .* exp.(1im*u[1].ϕ)
    s = (nodes.P + 1im*nodes.Q)/base_power
    mismatch = u_1 .* conj(LY_1*u_1) + s
    f = vcat(real(mismatch[2:end]), imag(mismatch[2:end]))
    err = maximum(abs.(f))
    f, err
end


function fund_jacobian(u, LY) 
    u_1 = u[1].v .* exp.(1im*u[1].ϕ)
    i_diag = spdiagm(sparse(LY[1] * u_1))
    u_1_diag = spdiagm(u_1)
    u_1_diag_norm = spdiagm(u_1./abs.(u_1))

    dSdϕ = 1im*u_1_diag*conj(i_diag - LY[1]*u_1_diag)
    dSdv = u_1_diag_norm*conj(i_diag) + u_1_diag*conj(LY[1]*u_1_diag_norm)

    # divide sub-matrices into real and imag part, cut off slack
    dPdϕ = real(dSdϕ[2:end, 2:end])
    dPdv = real(dSdv[2:end, 2:end])
    dQdϕ = imag(dSdϕ[2:end, 2:end])
    dQdv = imag(dSdv[2:end, 2:end])

    vcat(hcat(dPdϕ, dPdv), 
         hcat(dQdϕ, dQdv))
end


function update_state_vec!(J, x, f)
    x - J\f  # Newton-Raphson iteration
    # -> could try using NLsolve.jl
end


function update_fund_voltages!(u, x)
    u[1].ϕ[2:end] = x[1:(length(x)÷2)]
    u[1].v[2:end] = x[(length(x)÷2+1):end]
    u
end


function pf(nodes, settings, LY, plt_convergence = false)
    u = init_voltages(nodes, settings)
    x_f = fund_state_vec(u)
    f_f, err_f = fund_mismatch(nodes, u, LY, settings.base_power)

    n_iter_f = 0
    err_f_t = Dict()
    while err_f > settings.thresh_f && n_iter_f <= settings.max_iter_f
        J_f = fund_jacobian(u, LY)
        x_f = update_state_vec!(J_f, x_f, f_f)
        u = update_fund_voltages!(u, x_f)
        f_f, err_f = fund_mismatch(nodes, u, LY, settings.base_power)
        err_f_t[n_iter_f] = err_f
        n_iter_f += 1
    end

    println(u[1])
    if n_iter_f < settings.max_iter_f
        println("Fundamental power flow converged after ", n_iter_f, 
              " iterations (err < ", settings.thresh_f, ").")
    elseif n_iter_f == settings.max_iter_f
        println("Maximum of ", n_iter_f, " iterations reached.")
    end
    return u
end


# Harmonic Power Flow functions
"""Import Norton Equivalent parameters for all nonlinear devices in "nodes" from folder"""
function import_Norton_Equivalents(nodes, settings, folder_path="harmonic-power-flow\\Circuit Simulation\\")
    NE = Dict()
    nl_components = unique(nodes[nodes.type .== "nonlinear", "component"])
    for device in nl_components
        NE_df = CSV.read(folder_path * device * "_NE.csv", DataFrame)
        # transform to Complex type, enough to strip first paranthesis for successful parse
        vals = mapcols!(col -> parse.(ComplexF64, strip.(col, ['('])), NE_df[:, 3:end])
        NE_device = hcat(NE_df[:,1:2], vals)
        # filter columns for considered harmonics
        NE_device = NE_device[:, Between(begin, string(maximum(settings.harmonics)*settings.base_frequency))]
        # --> CHECK if this works as intended

        # change to pu system and choose if coupled 
        if settings.coupled
            I_N = Array(NE_device[NE_device.Parameter .== "I_N_c", 3:end])/settings.base_current
            LY_N_full = NE_device[NE_device.Parameter .== "Y_N_c", 2:end]
            LY_N = Array(LY_N_full[LY_N_full.Frequency .<= maximum(settings.harmonics)*settings.base_frequency, 2:end])/settings.base_admittance
        else
            I_N = Array(NE_device[NE_device.Parameter .== "I_N_uc", 3:end])/settings.base_current
            LY_N = Array(NE_device[NE_device.Parameter .== "Y_N_uc", 3:end])/settings.base_admittance
        end    
        NE[device] = [I_N, LY_N]
    end
    return NE
end

"""calculate the harmonic current injections at one node"""
function current_injections(nodes, nodeID, u, NE, harmonics)
    component = nodes[nodes.ID .== nodeID, "component"][1]
    I_N, LY_N = NE[component]
    # u as dict of dfs makes building this vector a bit complicated
    u_h = vcat([u[h][nodeID, "v"] .* exp.(1im*u[h][nodeID, "ϕ"]) for h in harmonics]...)
    # coupled: Y_N is a matrix, uncoupled: vector
    if size(LY_N)[1] > 1  # coupled case
        i_inj = vec(I_N) - vec(LY_N*u_h)
    else  # uncoupled case
        i_inj = vec(I_N) - spdiagm(vec(LY_N))*u_h
    end
    return i_inj
end

function current_balance(net, settings, u, LY, NE)
    # fundamental admittance matrix for nonlinear nodes
    LY_1_nl = LY[1][net.m:end,:]
    u_1 = u[1].v .* exp.(1im*u[1].ϕ)
    # fundamental line currents at nonlinear nodes
    dI_1 = LY_1_nl * u_1
    # harmonic admittance matrices as diagonal block matrix
    LY_h = blockdiag([LY[h] for h in settings.harmonics[2:end]]...)
    u_h = vcat([u[h][:, "v"] .* exp.(1im*u[h][:, "ϕ"]) for h in settings.harmonics[2:end]]...)
    dI_h = LY_h * u_h 

    # subtract the injected currents at each nonlinear node i
    for i in net.m:net.n
        i_inj = current_injections(net.nodes, net.nodes.ID[i], u, NE, settings.harmonics)
        dI_1[i-net.m+1] += i_inj[1]  # add injections at fundamental frequency...
        # ... and at all harmonic frequencies
        for p in 0:(settings.K-1)
            dI_h[p*net.n + i] += i_inj[p+2]
        end
    end
    vcat(dI_1, dI_h)
end


function harmonic_mismatch(net, settings, u, LY, NE)
    # fundamental power mismatch at linear buses except slack
    s = net.nodes.P[2:(net.m-1)]/settings.base_power + 1im*net.nodes.Q[2:(net.m-1)]/settings.base_power
    u_i = u[1][2:(net.m-1), "v"].*exp.(1im*u[1][2:(net.m-1), "ϕ"])
    u_j = u[1][:, "v"].*exp.(1im*u[1][:, "ϕ"])
    LY_ij = LY[1][2:(net.m-1), :]
    # power balance
    sl = u_i.*conj(LY_ij*u_j)  
    ds = s + sl  
    di = current_balance(net, settings, u, LY, NE) 
    # harmonic mismatch vector
    f_c = vcat(ds, di) 
    f = vcat(real(f_c), imag(f_c))
    err_h = maximum(abs.(f))  
    return f, err_h
end


function harmonic_state_vec(u, harmonics)
    xv = u[1].v[2:end]
    xϕ = u[1].ϕ[2:end]
    for h in harmonics[2:end]
        xv = vcat(xv, u[h].v)
        xϕ = vcat(xϕ, u[h].ϕ)
    end
    vcat(xv, xϕ)  # note: magnitude first
end   


function build_harmonic_jacobian(net, settings, u, LY, NE)
    m = net.m
    n = net.n
    K = settings.K

    u_vec = vcat([u[h].v .* exp.(1im*u[h].ϕ) for h in settings.harmonics]...)
    v_vec = vcat([u[h].v for h in settings.harmonics]...)
    u_diag = spdiagm(u_vec)
    u_norm = u_vec./v_vec
    u_norm_diag = spdiagm(u_norm)
    LY_diag = blockdiag([LY[h] for h in settings.harmonics]...)

    # construct Jacobian sub-matrices
    IV = LY_diag*u_norm_diag
    IT = 1im*LY_diag*u_diag

    # indices of first nonlinear bus at each harmonic
    nl_idx_start = m:n:n*(K+1)
    nl_idx_all = vcat([nl:(nl+n-m) for nl in nl_idx_start]...)
    u_nl = u_vec[nl_idx_all]
    v_nl = v_vec[nl_idx_all]
    u_nl_norm = u_nl./v_nl

    if settings.coupled
        # iterating through blocks vertically
        for h in 0:K
            # ... and horizontally
            for p in 0:K
                # iterating through nonlinear buses
                for i in m:n
                    # within NE "[2]" points to LY_N
                    LY_N = NE[net.nodes.component[i]][2]
                    # subtract derived current injections at respective idx
                    IV[h*n+i, p*n+i] -= LY_N[h+1, p+1]*u_nl_norm[(i-m+1)+p*(n-m+1)]
                    IT[h*n+i, p*n+i] -= 1im*LY_N[h+1, p+1]*u_nl[(i-m+1)+p*(n-m+1)]
                end
            end
        end
    else
        # iterating through blocks diagonally (p=h)
        for h in 0:K
            for i in m:n
                # LY_N is one-dimensional for uncoupled case
                LY_N = NE[net.nodes.component[i]][2]
                IV[h*n+i, p*n+i] -= LY_N[h+1]*u_nl_norm[(i-m+1)+h*(n-m+1)]
                IT[h*n+i, p*n+i] -= 1im*LY_N[h+1]*u_nl[(i-m+1)+h*(n-m+1)]
            end
        end
    end

    IV = IV[m:end, 2:end]  
    IT = IT[m:end, 2:end]  

    LY_1 = LY[1]
    u_1 = u[1].v .* exp.(1im*u[1].ϕ)
    i_diag = spdiagm(LY_1*u_1)
    u_diag = spdiagm(u_1)
    u_diag_norm = spdiagm(u_1./abs.(u_1))

    S1V1 = u_diag_norm*conj(i_diag) + u_diag*conj(LY_1*u_diag_norm)
    S1T1 = 1im*u_diag*(conj(i_diag - LY_1*u_diag))

    SV = hcat(S1V1[2:(m-1), 2:end], zeros(m-2, n*K))
    ST = hcat(S1T1[2:(m-1), 2:end], zeros(m-2, n*K))

    # combine all sub-matrices and return complete Jacobian
    vcat(hcat(real(SV), real(ST)),
         hcat(real(IV), real(IT)),
         hcat(imag(SV), imag(ST)),
         hcat(imag(IV), imag(IT)))
end


function update_harmonic_voltages!(u, x, harmonics, n)
    # slice x in half to separate voltage magnitude and phase
    xv = x[1:(length(x)÷2)]
    xϕ = x[(length(x)÷2+1):end]
    xϕ = xϕ .% (2*π)  # ensure phase smaller 2π
    for h in harmonics
        i = findall(harmonics .== h)[1] - 1
        if h == 1
            # update all nodes except slack at fundamental frequency
            u[h].v[2:end] = xv[1:n-1]
            u[h].ϕ[2:end] = xϕ[1:n-1]
        else
            # update all nodes at harmonic frequencies
            u[h].v = xv[i*n:((i+1)*n-1)]
            u[h].ϕ = xϕ[i*n:((i+1)*n-1)]
        end
    end
    return u
end


function hpf(net, settings)
    LY = admittance_matrices(net, settings.harmonics)
    u = pf(net.nodes, settings, LY)
    NE = import_Norton_Equivalents(net.nodes, settings)
    f, err_h = harmonic_mismatch(net, settings, u, LY, NE)
    x = harmonic_state_vec(u, settings.harmonics)
    n_iter_h = 0
    err_h_t = Dict()
    while err_h > settings.thresh_h && n_iter_h < settings.max_iter_h
        J = build_harmonic_jacobian(net, settings, u, LY, NE)
        x = update_state_vec!(J, x, f)
        u = update_harmonic_voltages!(u, x, settings.harmonics, net.n)  
        f, err_h = harmonic_mismatch(net, settings, u, LY, NE)
        err_h_t[n_iter_h] = err_h
        n_iter_h += 1
    end

    # getting rid of negative voltage magnitudes:
    for h in settings.harmonics
        u[h].ϕ[u[h].v .< 0] .+= π
        u[h].ϕ .= u[h].ϕ .% (2*π)
        u[h].v[u[h].v .< 0] = -u[h].v[u[h].v .< 0]
    end

    if n_iter_h < settings.max_iter_h
        println("Harmonic power flow converged after ", n_iter_h,
                " iterations (err < ", settings.thresh_h, ").")
    elseif n_iter_h == settings.max_iter_h
        println("Maximum of ", n_iter_h, " iterations reached. Harmonic power flow did not converge.")
    end
    return u, err_h, n_iter_h
end


function get_THD(u, nodes, harmonics)
    THD = DataFrame(THD_F = zeros(size(nodes)[1]), 
                    THD_R = zeros(size(nodes)[1]))
    for ID in nodes.ID
        THD[ID, "THD_F"] = sqrt(sum([u[h].v[ID]^2 for h in harmonics[2:end]]))./u[1].v[ID]
        THD[ID, "THD_R"] = sqrt(sum([u[h].v[ID]^2 for h in harmonics[2:end]]))./sqrt(sum([u[h].v[ID]^2 for h in harmonics]))
    end
end


settings1 = init_settings(true, [1, 3, 5])
net1 = init_power_grid(import_nodes_from_csv(NET_PATH*"net1"), import_lines_from_csv(NET_PATH*"net1"))
net2 = init_power_grid(import_nodes_from_csv(NET_PATH*"net2"), import_lines_from_csv(NET_PATH*"net2"))


@time u, err_h_final, n_iter_h = hpf(net2, settings1)
u[5]