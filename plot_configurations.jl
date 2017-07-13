println("Loading packages...")
using HDF5
using PyPlot
println("")

typealias Spin Int8
typealias Conf Array{Spin, 2}
typealias Confs Array{Spin, 3}

function plot_confs(confs::Confs, fname::String="", all::Bool=false)
    l = size(confs)[end]
    n = l
    if (l > 50) n = 50 end

    confidcs = floor(Int, linspace(1,l,n))

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

confs = h5read(ARGS[1], "configurations")

plot_confs(confs, ARGS[1])