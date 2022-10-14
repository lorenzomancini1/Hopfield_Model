# Standard Hopfield
## Critical Noise
We perturb a stored pattern with probability $p$ and we want to estimate the probability of recovering the original pattern. The activation is defined by a zero-temperature Monte Carlo
```{julia}
function monte_carlo(J::AbstractMatrix, σ::AbstractVector;
					nsweeps = 100, earlystop = 0, 
					β = 10, annealing = false)
```
where the ``earlystop = 0`` ensures that the dynamics stops if no more spin flips occur.
The estimation of the *Reconstruction Probability* is performed by running the MC for many samples and counting the rate of successes, i.e. if the events where $m = \sigma \cdot \xi^\mu/N$ is greater or equal than $0.95$.
```{julia}
function one_reconstruction_probability(N::Int, pp::AbstractVector, α, nsamples;
							nsweeps = 100, β = 10, earlystop = 0)
```
The function above computes the points of the reconstruction probability for a given $N$. 

