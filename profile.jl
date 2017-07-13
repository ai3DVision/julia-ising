import Base.dump
include("ising2d.jl")

using ProfileView

global const Tc = 1/(1/2*log(1+sqrt(2)))

p = Parameters()
p.L = 64
T = 0.5 # outside of code always T in units of Tc
p.n_sweeps = 2000
p.wolff = true

p.beta = 1/(T * Tc)
p.seed = 11235813
p.startconf = "ordered"
conf = p.startconf=="ordered"?ones(Spin, p.L, p.L):convert(Conf, rand([-1, 1], p.L, p.L));
build_neighbortable!(p);

Profile.clear()

@profile MonteCarlo(p)

ProfileView.view()