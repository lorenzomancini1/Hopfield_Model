module Hopfield_Model

using LinearAlgebra, Random, Statistics

include("StandardHopfield/standard_hopfield.jl")
export  SH

include("ModernHopfieldBinary/modern_hopfield_binary.jl")
export MHB

include("ModernHopfieldContinuous/modern_hopfield_gaussian.jl")
export MHC

end
