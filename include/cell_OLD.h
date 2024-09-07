#ifndef CELL_H
#define CELL_H

#include <vector>
#include "include/partial_OLD.h"

// https://stackoverflow.com/questions/7304511/partial-derivatives
// https://stackoverflow.com/questions/270408/is-it-better-in-c-to-pass-by-value-or-pass-by-reference-to-const

// using namespace std;
// marker-and-cell
// Material Derivative? https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8460516/

// implement partial, material derivatives, negative divergence
// advection

/*
Technically, we are using Euler equations rather than Navier-Stokes because we assume 0 viscosity

Replace label with enum? (air, fluid, solid)
    - How do we know that a cell is changing states (e.g. fluid becomes air)

Dirichlet boundary

-- Euler Momentum Equation --

Dot product
Material derivative
Design marker-and-cell
- Remember Dirichlet boundary for solid cells
Semi-Lagrangian Advection
    - Cubic interpolation - interpolate()
    - q^(n+1)=advect(u ⃗,∆t,q^n ) - advect()
    - Second-order Runge Kutta (page 45 of pdf)

-- Imcompressibility Condition --

Modified Incomplete Cholesky Conjugate Gradient Level Zero (MICGG(0))
    - PCG
        - applyPreconditioner() <- MIC(0)
        - See PCG pseudocode
    - Preconditioner
        - Modified Incomplete Cholesky Level 0: MIC(0)
            - May weight against Incomplete Cholesky Gradient but not necessary

Calculate timestep
    - Courant-Friedrichs-Lewy Condition

project() - CHAPTER 5
    - "make u divergence-free, enforce solid-wall boundary"
    - MIC(0)

Getters
    - u,v
    - grid cell width
    - label

Compute shaders (GLSL)

We can also constantly update itmestep based on u_max
    - Assume this is pretty computationally efficient (can do same O(N^2) pass)
*/

// move all function declarations into src cpp file

class Cell
{
public:
    Cell(float p, std::vector<float> vel);
    // General constructor

    // constructor for unmoving boundary cell
    Cell(float p);

    // copy constructor
    Cell(Cell &other);

    std::vector<std::vector<float>> getVelocities();

    // getters
    float getVelocity();
    float getDensity();
    cellType getLabel();

    // setters
    void setVelocity(float v);
    void setDensity(float rho);
    void setLabel(cellType cType);

    float p();    // p_i,j
    float &u();   // u_(i-1/2,j)
    float &v();   // u_(i,j-1/2)
    float &u_1(); // u_(i+1/2,j)
    float &v_1(); // u_(i,j+1/2)

private:
    float pressure;
    float density;
    std::vector<std::vector<float>> velocities;
    int rId;
    int cId;
    float velocity; // variables with normals?
    // shader - vertex and fragment
    // adjust opacity or create gradient basedd on velocity in shader

    void update_v(std::vector<float> &u_a, float dt, std::vector<float> g_vec);
    cellType label;

    enum cellType
    {
        FLUID = 1,
        SOLID = 2,
        AIR = 3
    };
}

#endif