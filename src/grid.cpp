#include "grid.h"
#include <array>
#include "cell.h"

Grid::Grid(int r, int c)
{
    // int items{r*c};
    std::vector<std::vector<Cell>> grid;
    std::vector<std::vector<float>> matrixA; // Adiag, Ax, Ay
    // grid.reserve(items);
    // matrixA.reserve(items * 3);

    for (int row = 0; row < r; row++)
    {
        for (int col = 0; col < c; col++)
        {
            grid.push_back(new Cell(0.0));
        }
    }
}
std::vector<std::vector<float>> Grid::&getMatrixA()
{
    return &matrixA;
}

std::vector<std::vector<Cell>> Grid::&getGrid()
{
    return grid;
}

float Grid::get_dx()
{
    return dx;
}

float Grid::get_rho()
{
    return rho;
}
cellType Grid::label(int i, int j)
{
    // auto cell_obj = getGrid();

    return getCell(i, j)->getLabel();
}

Cell Grid::&getCell(int i, int j)
{
    return &grid[i][j];
}

// Matrix A functions for preconditioner

float &Grid::Adiag(int i, int j)
{
    // check what matrix A does
    return &matrixA[i][j][0];
}

float &Grid::Ax(int i, int j)
{
    return &matrixA[i][j][1];
}
float &Grid::Ay(int i, int j)
{
    return &matrixA[i][j][2];
}

float Grid::getAdiag(i, j)
{
    return matrixA[i][j][0];
}

float Grid::getAx(i, j)
{
    return matrixA[i][j][1];
}

float Grid::getAy(i, j)
{
    return matrixA[i][j][2];
}