#include <iostream>
#include <eigen3/Eigen/Eigen>

#include "condition_functions.h"

int main()
{
    
    int N_x = 4;
    int N_y = 4;
    double dx = 1.0/N_x;
    double dy = 1.0/N_y;
    double dt = 0.025;

    double startX = 0.0;
    double endX = 1.0;
    double startY = 0.0;
    double endY = 1.0;
    double startT = 0.0;
    double endT = 1.0/2.0;

    int noOfTimePoints = 0;
    double timeCounter = 0.0;

    while(timeCounter <= (endT - startT))
    {
        timeCounter += dt;
        noOfTimePoints++;
    }

    Eigen::VectorXd xSpacePoints, ySpacePoints, timePoints;
    xSpacePoints = Eigen::VectorXd::LinSpaced(N_x + 1, startX, endX);
    ySpacePoints = Eigen::VectorXd::LinSpaced(N_y + 1, startY, endY);
    timePoints = Eigen::VectorXd::LinSpaced(noOfTimePoints, startT, endT);

    // Solution matrix for the 0th timestep
    Eigen::MatrixXd U(ySpacePoints.size(), xSpacePoints.size());
    U.setZero();

    // Solution matrix for the 1st timestep
    Eigen::MatrixXd V(ySpacePoints.size(), xSpacePoints.size());
    V.setZero();

    // IC

    for(int k = 0; k < U.rows(); k++)
    {
        for(int j = 0; j < U.cols(); j++)
        {
            U(k, j) = ic_func(xSpacePoints(j, 0), ySpacePoints(k, 0));
        }
    }

    // BCs

    // x boundaries
    for(int k = 0; k < U.rows(); k++)
    {
        V(k, 0) = U(k, 0) = x_lhs_dirichlet_bc_func(ySpacePoints(k, 0));
        V(k, V.cols() - 1) = U(k, U.cols() - 1) = x_rhs_dirichlet_bc_func(ySpacePoints(k, 0));
    }

    // y boundaries
    for(int j = 0; j < U.cols(); j++)
    {
        V(0, j) = U(0, j) = y_lower_dirichlet_bc_func(xSpacePoints(j, 0));
        V(V.rows() - 1, j) = U(U.rows() - 1, j) = y_upper_dirichlet_bc_func(xSpacePoints(j, 0));
    }

}
