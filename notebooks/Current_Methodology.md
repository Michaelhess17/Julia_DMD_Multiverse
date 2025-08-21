# Current Methodology
## Phaser
### Estimate phase
### Interpolate each cycle to 100 time steps

## Delay Embedding
### Delay embed more than one cycle (e.g. tau = 1, k = 150)

## Projection matrix
### Make an orthonormal matrix of complex sinusoids at varying frequencies
#### Currently just using frequencies at intervals of 0.5 gait cycle$^{-1}$ from 0.0 to ~20 gait cycle$^{-1}$
#### Use QR decomposition to make the matrix orthornormal
#### Align the sinuosoids frequencies/lengths to the delay embedding size/length

## Project data onto basis
## Fit DMD model
## Use conjugate transpose of matrix to go back to original (time delayed) state space
