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

    // A matrix
    void Adiag(int i, int j);
    void Ax(int i, int j);
    void Ay(int i, int j);

private:
    std::vector<std::vector<float>> matrixA;
    float dx;
    float density;
    std::vector<std::vector<Cell>> grid;
    std::vector<std::vector<float>> matrixA;
}

#endif