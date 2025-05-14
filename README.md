# Julia_DMD_Multiverse
Workspace for me to implement DMD variants from the multiverse before porting them to DataDrivenDMD

Most of the code here is based on either the [PyDMD](https://github.com/PyDMD/PyDMD) scripts or Dr. Matthew Colbrook's scripts from [his DMD Multiverse GitHub Repo](https://github.com/mcolbrook/DMD-Multiverse/)

## Completed (tested)
- Exact DMD (via pseudo-inverse) (`src/exactDMD.jl`)
- Bagged, optimized DMD (`src/bopdmd.jl`)
- Residual DMD (via QR decomposition) (`src/residualDMD.jl`)
- Deep Learning DMD (with penalties based on DMD prediction accuracy in the embedded space, reconstruction of data with the autoencoder, linearity of the DMD fit/residuals, and L1/L2) (`src/deepLearningDMD.jl`)

## Translated (no testing)
- Randomized DMD (`src/randomizedDMD.jl`)

## In Progress (unusable)
- Compressed Sensing DMD (`in_progress/compressedSensingDMD.jl`)
- Kernelized Residual DMD (`in_progress/kernel_resDMD.jl`)

## Interested/Not Yet
- HAVOK
- DMD with Control
- Measure-preserving (E)DMD

## Not Planned (implemented in DataDrivenDMD)
- Total Least Squares DMD
- Forward Backward DMD
