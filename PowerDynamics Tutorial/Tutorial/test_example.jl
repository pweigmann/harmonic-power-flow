# in the REPL do: ] to enter the Package manager
#  and add PowerDynamics, Plots, LaTeXStrings,

using Pkg
Pkg.instantiate()
using PowerDynamics
#using Revise

node_list=[]
    append!(node_list, [SlackAlgebraic(U=1.)])
    append!(node_list, [SwingEqLVS(H=5., P=1., D=0.1, Ω=50,Γ=0.1,V=1.)])
    append!(node_list, [SwingEqLVS(H=5., P=1., D=0.1, Ω=50,Γ=0.1,V=1.)])
    append!(node_list, [PQAlgebraic(P=-1,Q=-1)])

line_list=[]
    append!(line_list,[StaticLine(from=1,to=2,Y=-1im/0.02)])
    append!(line_list,[StaticLine(from=2,to=3,Y=-1im/0.02)])
    append!(line_list,[StaticLine(from=3,to=4,Y=-1im/0.02)])
    append!(line_list,[StaticLine(from=2,to=4,Y=-1im/0.02)])

powergrid = PowerGrid(node_list,line_list)

operationpoint = find_operationpoint(powergrid)


result = simulate(Perturbation(2, :ω, Inc(0.1)), powergrid, operationpoint, timespan = (0.0,10))

include("plotting.jl")
plot_res(result,powergrid,2)

result2 = simulate(LineFault(2,4, ), powergrid, operationpoint, timespan = (0.0,10))

plot_res(result2,powergrid,2)
