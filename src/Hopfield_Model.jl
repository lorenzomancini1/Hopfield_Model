module Hopfield_Model

include("StandardHopfield/standard_hopfield.jl")
export  SH

include("ModernHopfield/binary/modern_hopfield_binary.jl")
export MHB

include("ModernHopfield/continuous/continuous_hopfield.jl")
export MHC

end
