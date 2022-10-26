## Critical Noise
How much information does the model need to reconstruct correctly a perturbed pattern?
This quantity is something related to the Critical Noise, i.e. that perturb probability $p_c$ above which the model is unable to recover an original pattern.
One can also think that this measure tells us how large the basins of attractions are.

We estimate $p_c$ only for finite sizes and we infer the value of the thermodynamic limit just performing a finite-size scaling analysis.
The main quantity that we need for this is the *reconstruction frequency* or *reconstruction probability*; in particular the following function (from `reconstruction_probability.jl`):

- `functio one_reconstruction_probability(N::Int, pp::AbstractVector, α, nsamples;...)` 
    1. generates an Hopfield Sample (i.e. the patterns $\xi$ and the associated matrix $J$);
    2. for each sample it perturbs a random pattern with noise $p$ and tries to recover the original one;
    3. if the final overlap is $\geq 0.8$ we consider it a success; 
    4. repeat 1-3 `nsamples` times and perform an average.

The `function reconstruction_probability(NN::AbstractVector,
    α;...)` is essentially the same repeated for different sizes.

