#ifndef GRID_H
#define GRID_H

#include "cell.h"
#include <array>

class Grid
{
public:
    Grid(int r, int c);
    std::vector<std::vector<float>> &getMatrixA();
    float get_dx();
    float get_rho();
    cellType label(int i, int j);
    std::vector<std::vector<Cell>> &getGrid();
    Cell &getCell(int i, int j);

    // Operations on matrix A
    // These three functions: 77
    // Index of each pos: adiag(0), ax(1), ay(2)
    float &Adiag(int i, int j); // respective A diag
    float &Ax(int i, int j);    // positive x neighbor
    float &Ay(int i, int j);    // positive y neighbor

    float getAx(i, j); // Aplusi
    float getAy(i, j); // Aplusj

private:
    std::vector<std::vector<float>> matrixA;
    float dx;
    float density;
    std::vector<std::vector<Cell>> grid;
}

#endif