# Eulerian Fluid Simulation

Currently in progress.

## Approach

We need to properly test a 2D fluid simulation before upgrading to a 3D one.

### We follow the following algorithm for a 2D fluid simulation

- Determine timestep (Using Courant–Friedrichs–Lewy condition)
- Advect velocity vector
  - Semi-Lagrangian Advection
    - Retrieve current value of quantity q
- Update velocity with external force(s) (gravity)
- Project velocity
  - Calculate negative divergence in function rhs(), store in vector b
  - Set up entries of coefficient matrix A
  - We perform Modified Incomplete Cholseky Level Zero to get our preconditioner
    - MIC(0) and the preconditioner serve to optimize computational speed/efficiency
  - Use our given matrix A and vector b, get pressure vector via Preconditioned Conjugate Gradient
  - Use new pressure values to update velocity
