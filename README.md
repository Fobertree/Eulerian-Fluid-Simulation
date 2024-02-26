# Eulerian Fluid Simulation

Currently in progress.

## Basics/Motivation

We are motivated by two equations (Navier-Stokes)

- $\nabla \bullet \vec{u}=0$ (incompressibility condition)
  - Can be thought of in terms of conservation of mass/Newton's First Law.
- $\frac{D\vec{u}}{Dt} = -\frac{1}{p}\nabla p+\vec{g} + v\nabla \bullet \nabla \vec{u}$
  - Can be thought of in terms of $\sum F=ma$

In this simulation, I drop viscosity, meaning I am using the Euler equations instead.

## Approach

We need to properly test a 2D fluid simulation before upgrading to a 3D one.

### We follow the following algorithm for a 2D fluid simulation

- Determine timestep (Using Courant–Friedrichs–Lewy condition)
- Advect velocity vector
  - Semi-Lagrangian Advection
    - Retrieve current value of quantity q
      - Interpolation
- Update velocity with external force(s) (gravity) TO-DO
- Project velocity (incompressibility condition) TO-DO
  - Calculate negative divergence in function rhs(), store in vector b
  - Set up entries of coefficient matrix A
  - We perform Modified Incomplete Cholesky Level Zero to get our preconditioner
    - MIC(0) and the preconditioner serve to optimize computational speed/efficiency
    - Note: The Cholesky Decomposition is sometimes intuitively understood as the "square root of a matrix"
  - Use our given matrix A and vector b, get pressure vector via Preconditioned Conjugate Gradient
  - Use new pressure values to update velocity

Misc.

Add usolid(i,j)

![Whiteboard Outline](./Images/IMG_2745.jpg)
