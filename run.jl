for L in [100]
	for T in [0.2 0.8 1.0 1.2 5]
       run(`julia ising2d.jl $(L) $(T) 10000`)
    end
end