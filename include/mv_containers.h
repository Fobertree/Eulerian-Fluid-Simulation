//
// Created by Alex on 9/28/2024.
//

#ifndef MATVEC_CONTAINERS_H
#define MATVEC_CONTAINERS_H

#include <Eigen/Core>
#include <string>
#include <iostream>

typedef Eigen::VectorXd vec_d;
//typedef Eigen::MatrixX2d mat2;
typedef Eigen::MatrixX<double> mat_d;
/*
 * Add boundary checks
 *
 *
 */

// Switch all vec_d to vec_d container if we encounter boundary check errors

template<typename T = double>
struct vec_d_container {
    // 1d vector

    vec_d data;
    size_t length;
    std::string name = "None";
    // replace below with smth more solid later
    size_t r_size {0}, c_size {0};

    vec_d_container<T>(const vec_d& data, size_t& length, const size_t& rows, const size_t& cols, std::string&& name = "None")
    {
        this-> data = data;
        this-> length = length;
        this-> name = name;
        r_size = rows;
        c_size = cols;
    }

    T& at(int i)
    {
        if (i >= length) {
            // flush immediately because error
            std::cerr << "::BOUNDS ERROR::" << name << ": indexing length: " << length << " with index: " << i
                      << std::endl;
            return nullptr;
        }

        return data(i);
    }

    T& at(int i, int j)
    {
        if (i >= r_size || i < 0)
        {
            std::cerr << "::BOUNDS ERROR::" << name << ": indexing with # rows being: " << r_size << " with index: " << i
                      << std::endl;
            return nullptr;
        }

        if (j >= c_size || j < 0)
        {
            std::cerr << "::BOUNDS ERROR::" << name << ": indexing with # cols being: " << c_size << " with index: " << j
                      << std::endl;
            return nullptr;
        }

        size_t idx = i * c_size + j;
        if (idx >= length) {
            // flush immediately because error
            std::cerr << "::BOUNDS ERROR::" << name << ": indexing length: " << length << " with index: " << idx
                      << std::endl;
            return nullptr;
        }

        return data(static_cast<int>(idx));
    }

    int operator()(int i, int j)
    {
        if (this->at(i,j) != nullptr)
        {
            return 0;
        }
        // error code
        return 1;
    }

    int operator()(int i)
    {
        if (this->at(i) != nullptr)
        {
            return 0;
        }
        // error code
        return 1;
    }

    void move(vec_d&& other)
    {
        data = other;
    }
};

template<typename T = double>
struct mat_d_container {
    // nd matrix
    mat_d data;

    T& at()
    {

    }

    void move(mat_d&& other)
    {
        data = other;
    }
};

typedef struct vec_d_container<> vec_DD;
typedef struct mat_d_container<> mat_DD;


#endif //MATVEC_CONTAINERS_H
