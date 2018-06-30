#ifndef GUARD_PDESolver_h
#define GUARD_PDESolver_h

#include <iostream>
#include <eigen3/Eigen/Eigen>
#include <vector>
#include <string>
#include <fstream>

#include "condition_functions.h"

class PDESolver
{
    public:
        PDESolver(
                int,
                int,
                double,
                double,
                double,
                double,
                double,
                double,
                double,
                std::string,
                std::string
        );

        PDESolver(
                int,
                int,
                double,
                double,
                double,
                double,
                double,
                double,
                double,
                std::string,
                std::string,
                std::string
        );

        void initialise_vectors_matrices();
        void create_discretisation_vectors();
        void create_solution_matrices();
        void perform_mathematical_routines();
        void apply_prelim_ic_bcs();
        void create_kronecker_product_matrices();
        void solve_pde();
        void create_data_file(std::string);
        void plot_solution(std::string);

    private:
        const int N_x;
        const int N_y;
        const double dx;
        const double dy;
        const double dt;

        const double startX;
        const double endX;
        const double startY;
        const double endY;
        const double startT;
        const double endT;

        const std::string xBCType;
        const std::string yBCType;
        const std::string neumannBCScheme;

        Eigen::VectorXd xSpacePoints;
        Eigen::VectorXd ySpacePoints;
        Eigen::VectorXd timePoints;

        Eigen::MatrixXd U;
        Eigen::MatrixXd nextTimestepMatrix;

        int xMatricesDim;
        int yMatricesDim;

        Eigen::MatrixXd kroneckerProdMatrixA;
        Eigen::MatrixXd kroneckerProdMatrixB;

        std::vector<Eigen::MatrixXd> solutionMatrices;
};

#endif //GUARD_PDESolver_h
