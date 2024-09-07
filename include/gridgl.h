//
// Created by Alexa on 8/31/2024.
//

// honestly don't even think this grid gen is necessary for our Fluid Sim but I'll see later

#ifndef GRID_GRID_H
#define GRID_GRID_H

#include <vector>
#include <cstring>
#include <utility>

class Grid
{
public:
    [[nodiscard]] static float normX(float i)
    {
        // norm between -1 and 1
        PRINT("X: " << i);
        i = (i-0.5f) * 2.0f;
        //i = (i / _scr_width) * 2.0f - 1.0f;
        return i;
    }

    [[nodiscard]] static float normY(float i)
    {
        // norm between -1 and 1
        i = (i-0.5f) * 2.0f;
        //i = (i / _scr_height) * 2.0f - 1.0f;
        return i;
    }

    Grid(float width, float height, float scr_width, float scr_height)
    {
        _width = width, _height = height, _scr_width = scr_width, _scr_height = scr_height;
        numHeight = static_cast<int>(scr_height/height);
        numWidth = static_cast<int>(scr_width / width);
        total = numHeight * numWidth;

        if (total == 0)
        {
            std::cout << "Failure to initialize grid, size 0" << std::endl;
            exit(1);
        }

        generate_grid();
    }


    void generate_grid()
    {
        vertices.reserve(total * 6 * 3);
        ebo_indices.reserve(total * 6);

        float xstep = _width / _scr_width, ystep = _height / _scr_height;

        unsigned int vertexIndex{0};

        for (int y_idx = 0; y_idx < numHeight; y_idx++)
        {
            float y = (float)y_idx * ystep;
            for (int x_idx = 0; x_idx < numWidth; x_idx++)
            {
                float x = (float)x_idx * xstep;
                // push set of vertices. Two triangles (6 points)
                // OpenGL expects counterclockwise order of triangle vertices

                // bottom left
                vertices.insert(vertices.end(),{normX(x), normY(y), 0.0f}); // bottom left
                vertices.insert(vertices.end(),{normX(x + xstep), normY(y), 0.0f}); // bottom right
                vertices.insert(vertices.end(),{normX(x), normY(y + ystep), 0.0f}); // top left
                // top right
                vertices.insert(vertices.end(),{normX(x + xstep),normY(y),0.0f}); // bottom right
                vertices.insert(vertices.end(),{normX(x + xstep),normY(y + ystep), 0.0f}); // top right
                vertices.insert(vertices.end(),{normX(x), normY(y + ystep), 0.0f}); // top left

                // EBO for quad (double dorito)
                ebo_indices.insert(ebo_indices.end(), {
                    vertexIndex, vertexIndex + 1, vertexIndex + 2, // 1st dorito
                    vertexIndex + 1, vertexIndex + 4, vertexIndex + 2 // 2nd dorito
                });

                vertexIndex += 6;
            }
        }

        std::cout << "vertices size: " << vertices.size() << std::endl;
        std::cout << "ebo size: " << ebo_indices.size() << std::endl;
    }

    [[nodiscard]] inline const std::vector<float>& getVertices()
    {
        // vertices is type vector<float>. Return as float* for float[]
        for (auto &i : vertices)
        {
            PRINT(i);
        }
        return vertices;
    }

    [[nodiscard]] inline const std::vector<unsigned int>& getEBO()
    {
        // vertices is type vector<float>. Return as float* for float[]
        return ebo_indices;
    }

    [[nodiscard]] inline float getHeight() const
    {
        return _height;
    }

    [[nodiscard]] inline float getWidth() const
    {
        return _width;
    }

    [[nodiscard]] inline float getNormHeight() const
    {
        return normY(_height / _scr_height);
    }

    [[nodiscard]] inline float getNormWidth() const
    {
        return normX(_width / _scr_width);
    }
private:
    float _width, _height, _scr_width, _scr_height;
    size_t numHeight, numWidth, total{0};

    std::vector<unsigned int> ebo_indices;
    std::vector<float> vertices;
};

#endif //GRID_GRID_H
