# cheatsheet Julia DataFrames
using DataFrames

df = DataFrame(A = 1:5, B = [4,7,8,9,4])

# harmonic Voltage df, comparing two approaches
harmonics = [1,3,5]
V1 = DataFrame(harmonic = 1, ID = [1,2,3,4], V_m = 1, V_a = 0)
V2 = Dict(1 => DataFrame(ID = [1,2,3,4], V_m = 1, V_a = 0))

for h in harmonics[2:end]
    V1 = vcat(V1, DataFrame(harmonic = h, ID = [1,2,3,4], V_m = 0.1, V_a = 0))
    V2[h] = DataFrame(ID = [1,2,3,4], V_m = 0.1, V_a = 0)
end

V1  # singular DataFrame with harmonic specified in column "harmonic"
V2  # Dict of DataFrames, key = harmonic


# usage examples
# construct voltage for one harmonic as vector
V1_1 = filter!(row -> row.harmonic == 1, V1)
V1_vec = V1_1.V_m.*exp.(1im.*V1_1.V_a)

V2_vec = V2[1].V_m.*exp.(1im.*V2[1].V_a)

V1_vec == V2_vec