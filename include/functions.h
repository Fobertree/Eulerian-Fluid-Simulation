#ifndef FUNCTIONS_H
#define FUNCTIONS_H
typedef std ::vector<std::vector<float>> vvf;
typedef std ::vector<float> vf;

#include <vector>
#include "grid.h"

// declare all functions
float second_rk(std::vector<float> x_g, std::vector u_g, float dt); // 2nd order Runge-Kutta
std::vector<float> project(float dt, Cell cell);                  // DOING
void scalar_mult(std::vector<float> &vec, float scalar);
std::vector<float> linear_combination(vf v1, vf v2, float s1 = 1, float s2 = 1);
template <class T>
std::vector<std::vector<float>> matrixMult(T matrix_a, T matrix_b); // CHECK
constexpr auto applyA = &matrixMult;
std::vector<float> scalar_mult(std::vector<float> vec, float scalar);
std::vector vec_subtract(std::vector<float> vec1, std::vector<float> vec2);
float linear_interpolate(std::vector<float> x_p, float dx, std::vector<std::vector<Cell>> &grid);

// --TO-DO--?
float cubic_interpolate(); // TO DO (low priority since we already have linear interpolation)
float mic_zero();          // TO DO
float dot_product(std::vector<float> v1, std::vector<float> v2);
float vectorNorm(std::vector<float>); // defaulting to euclidean norm but may change
float advect(u_vec, dt, q);
void applyPreconditioner(std::vector<std::vector<float>> preconditioner, std::vector<float> vector);                        // one argument = preconditioner (from MIC(0) or IC). PCG
std::vector<float> semiLagrangian(std::vector<float> x_g, std::vector u_g, float dt, std::vector<std::vector<Cell>> &grid); // IN PROGRESS
void checkTimestep(float &dt, float u_max, float dx, int cfl_num);
float updatePressure(std::vector<float> u_n, float dt, float density, float dx, float p_next, float p_low); // DELETE?
float updateVelocity(Grid &grid, float dt, float density, float dx);                                        // update velocity based on rhs and pressure. IN PROGRESS
float rhs(float dt, float density, float dx, float p_next, float p_low);
float rhs(Cell cell);
std::vector<std::vector<float>> getPreconditioner(Grid &grid);                                         // setup A matrix
std::vector<std::vector<float>> get_xq_j_quantities(Grid &grid, std::vector<float> x_p);               // DELETE?
std::vector<std::vector<float>> setupA(std::vector<std::vector<Cell>> &grid, float dx, float density); // CHECK
std::vector<std::vector<float>> fowardSub(std::vector<std::vector> A, std::vector<float> b);
vf pcg(std::vector<std::vector<float>> A, std::vector<float> r, Grid &grid);
void addGravity(std::vector<float> &u_a, float dt, float g);
// TODO:
std::vector<float> usolid(i, j);
std::vector<float> vsolid(i, j);

// std::vector<std::vector<float>> setupA
/*
Math

LU factorization??
Modified Incomplete Cholesky Level 0
Dot product
Divergence
vectorNorm (Euclidean)

advect(u_vec,dt,q)
applyPreconditioner <- PCG
Interpolate <- semi-lagrangian
InterpolateAvg
checkTimestep (CFL)
pressureUpdate
applyA <- can we template a matrix multiplicaiton or smth. template alias
https://stackoverflow.com/questions/72463668/alias-function-template
getPreconditioner
setupA
x_j, x_j+1, q_j, q_j+1 getter for interpolation
*/

/*

PCG Steps

 Set initial guess p = 0 and residual vector r = b (If r = 0 then
return p)
• Set auxiliary vector z = applyPreconditioner(r), and search
vector s = z
• σ = dotproduct(z, r)
• Loop until done (or maximum iterations exceeded):
• Set auxiliary vector z = applyA(s)
• α = σ/dotproduct(z,s)
• Update p ← p + αs and r ← r − αz
• If max |r| ≤ tol then return p
• Set auxiliary vector z = applyPreconditioner(r)
• σnew = dotproduct(z, r)
• β = σnew/σ
• Set search vector s = z + βs
• σ = σnew
• Return p (and report iteration limit exceeded)
*/

#endif