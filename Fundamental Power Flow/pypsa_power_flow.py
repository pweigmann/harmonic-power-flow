from __future__ import print_function, division
import pypsa
import numpy as np

network = pypsa.Network()

# add buses
network.add("Bus", "Bus 1", v_nom=0.23)
network.add("Bus", "Bus 2", v_nom=0.23)
network.add("Bus", "Bus 3", v_nom=0.23)
network.add("Bus", "Bus 4", v_nom=0.23)

print(network.buses)

network.add("Line", "Line 1", bus0="Bus 1", bus1="Bus 2", r=0.5, x=0.5)
network.add("Line", "Line 2", bus0="Bus 2", bus1="Bus 3", r=1, x=4)
network.add("Line", "Line 3", bus0="Bus 3", bus1="Bus 4", r=0.5, x=1)
network.add("Line", "Line 4", bus0="Bus 4", bus1="Bus 1", r=0.5, x=1)

print(network.lines)

network.add("Generator", "slack",
            bus="Bus 1",
            control="Slack")
network.add("Generator", "Gen2",
            bus="Bus 2",
            p_set=0.0002,
            control="PV")

# add a load at bus 1
network.add("Load", "nonlinear load",
            bus="Bus 4",
            p_set=0.00025,
            q_set=0.0001)
network.add("Load", "linear load",
            bus="Bus 3",
            p_set=0.0001,
            q_set=0.0001)

network.pf()

print(network.lines_t.p0)
print(network.buses_t.v_ang)
print(network.buses_t.v_mag_pu)
