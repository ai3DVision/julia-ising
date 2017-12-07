# L T n_sweeps wolff

println("Loading packages...")
using HDF5
using PyPlot
println("")

const Spin = Int8
const Conf = Array{Spin, 2}
const Confs = Array{Spin, 3}

global const Tc = 1/(1/2*log(1+sqrt(2)))

type Parameters
    L::Int
    beta::Float64
    n_sweeps::Int
    seed::Int
    startconf::String # random, ordered

    wolff::Bool # false: metropolis, true: wolff

    neigh::Array{Int, 3}
    Parameters() = new()
end

type Measurements
    configurations::Confs # first two dim = conf, third dim = n_conf
    energies::Array{Float64, 1}
    magnetization::Array{Float64, 1}
    Measurements() = new()
end


function main(ARGS::Array{String})
    p = Parameters()
    p.L = 20
    T = 1.2 # outside of code always T in units of Tc
    p.n_sweeps = 50000
    p.wolff = false

    if length(ARGS) > 0
        p.L = parse(Int, ARGS[1])
        T = parse(Float64, ARGS[2])
        p.n_sweeps = parse(Int, ARGS[3])

        if length(ARGS) > 3
            p.wolff = (lowercase(ARGS[4]) == "wolff")
        end
    end

    println("L = ", p.L)
    println("T = ", T)
    println("N_SWEEPS = ", p.n_sweeps)
    println("WOLFF = ", p.wolff)

    p.beta = 1/(T * Tc)
    p.seed = 11235813
    p.startconf = "ordered"

    println("MONTECARLO...")
    m = MonteCarlo(p);
    println("done.\n")

    println("DUMPING...")
    dump(p,m)
    println("done.")

    if length(ARGS) == 0
        println("PLOTTING...")
        plot_confs(m.configurations, "confs")
        println("done.")
    end

end


function MonteCarlo(p::Parameters)  
    
    build_neighbortable!(p)
    srand(p.seed)

    m = Measurements()
    m.configurations = Array{Spin, 3}(p.L, p.L, p.n_sweeps+1);
    m.energies = Array{Float64, 1}(p.n_sweeps+1)
    m.magnetization = Array{Float64, 1}(p.n_sweeps+1)

    conf = p.startconf=="ordered"?ones(Spin, p.L, p.L):convert(Conf, rand([-1, 1], p.L, p.L))
    energy::Float64 = calc_energy(p, conf)
    m.configurations[:,:,1] = conf[:,:]
    m.energies[1] = energy
    m.magnetization[1] = sum(conf)
    
    tic()

    acc_rate = 0.
    @inbounds for s in 1:p.n_sweeps

        if !p.wolff
            dE, drate = sweep_Metropolis(p, conf);
            energy += dE
            acc_rate += drate
        else
            drate = sweep_WolffCluster(p, conf);
            energy = calc_energy(p, conf)
            acc_rate += drate
        end
        
        m.configurations[:,:,s+1] = conf
        m.energies[s+1] = energy
        m.magnetization[s+1] = sum(conf)

        if mod(s, 1000)==0
            @printf("\t%d (acc_rate: %.2f%%)\n", s, acc_rate/10)
            acc_rate = 0.
        end
        
        
    end

    toc()

    return m
end

function sweep_Metropolis(p::Parameters, conf::Conf)

    spinflips = 0
    dE_sum = 0.
    @inbounds for x in 1:p.L
        @inbounds for y in 1:p.L

            # Metropolis
            dE = 2. * conf[x,y] * sum(conf[p.neigh[x,y,:]])

            if dE <= 0 || rand(1)[1] < exp(- p.beta*dE)
                conf[x,y] *= -1
                spinflips += 1
                dE_sum += dE
            end

        end
    end

    return dE_sum, spinflips/(p.L^2)
end

function sweep_WolffCluster(p::Parameters, conf::Conf)
    const N = Int(p.L^2)

    cluster = Array{Int, 1}()
    tocheck = Array{Int, 1}()

    s = rand(1:N)
    push!(tocheck, s)
    push!(cluster, s)

    while !isempty(tocheck)
        cur = pop!(tocheck)
        @inbounds for n in p.neigh[cur:N:end]

            @inbounds if conf[cur] == conf[n] && !(n in cluster) && rand() < (1 - exp(- 2.0 * p.beta))
                push!(tocheck, n)
                push!(cluster, n)
            end

        end
    end

    for spin in cluster
        conf[spin] *= -1
    end

    return length(cluster)/N
end

function build_neighbortable!(p::Parameters)
    lattice = reshape(1:p.L^2, (p.L, p.L))
    p.neigh = Array{Int, 3}(p.L, p.L, 4)
    p.neigh[:,:,1] = circshift(lattice, [1, 0]) #up
    p.neigh[:,:,2] = circshift(lattice, [0, -1]) #right
    p.neigh[:,:,3] = circshift(lattice, [-1, 0]) #down
    p.neigh[:,:,4] = circshift(lattice, [0, 1]) #left
end

# function calc_energy(conf::Conf)::Float64
#     return - sum(conf .* circshift(conf, [0,1]) + conf .* circshift(conf, [1,0]))
# end

function calc_energy(p::Parameters, conf::Conf)::Float64
    E = 0.0
    @simd for x in 1:p.L
        @simd for y in 1:p.L
            @inbounds E += - (conf[x,y]*conf[p.neigh[x,y,1]] + conf[x,y]*conf[p.neigh[x,y,2]])
        end
    end
    return E
end


function dump(p::Parameters, m::Measurements)
    fname = "Ising2D_L$(p.L)_T$(round((1/p.beta)/Tc, 2)).h5"
    if p.wolff
        fname = "Ising2D_L$(p.L)_T$(round((1/p.beta)/Tc, 2))_wolff.h5"
    end
    if isfile(fname) warn("Overwriting existing results...") end
    f = h5open(fname, "w")
    f["configurations"] = m.configurations
    f["energy"] = m.energies
    f["magnetization"] = m.magnetization
    f["T"] = (1/p.beta)/Tc
    f["SEED"] = p.seed
    f["L"] = p.L
    f["WOLFF"] = Int(p.wolff)
    close(f)
end

function plot_conf(c::Conf, fname::String="")
    figure(figsize=(3,3))
    ax = gca()
    imshow(c, cmap="gray")
    ax[:set_xticks]([])
    ax[:set_yticks]([])
    if fname!=""
        savefig(fname*".pdf")
    end
end

function plot_confs(confs::Confs, fname::String="", all::Bool=false)
    l = size(confs)[end]
    n = l
    if (l > 50) n = 50 end

    confidcs = floor.(Int, linspace(1,l,n))

    n_rows = Int(ceil(n/5))
    n_cols = n<5?n:5

    fig, axes = subplots(n_rows, n_cols, figsize=(3*n_cols,3*n_rows))
    for r in 1:n_rows
        for c in 1:n_cols
            linidx = (r-1)*n_cols+c
            if linidx<=n
                axes[r,c][:imshow](confs[:,:,confidcs[linidx]], cmap="gray")
                axes[r,c][:set_title]("$(confidcs[linidx])")
            else
                axes[r,c][:set_frame_on](false)
            end
            axes[r,c][:set_xticks]([])
            axes[r,c][:set_yticks]([])
        end
    end

    tight_layout()

    if fname!=""
        savefig(fname*".pdf")
    end
end

main()