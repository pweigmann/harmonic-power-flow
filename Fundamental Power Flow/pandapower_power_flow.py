import pandapower as pp

net = pp.create_empty_network("mini")

pp.create_buses(net, 3, 20, index=[0, 1, 2])
pp.create_line(net, 0, 1, 0.01, std_type="NAYY 4x50 SE")
pp.create_line(net, 1, 2, 0.01, std_type="NAYY 4x50 SE")
pp.create_line(net, 2, 0, 0.01, std_type="NAYY 4x50 SE")

pp.create_load(net, bus=0, p_mw=100, q_mvar=100)
pp.create_gen(net, bus=1, p_mw=-100, slack=True)

pp.runpp(net)

net.res_bus
