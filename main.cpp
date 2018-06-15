#include <iostream>
#include <eigen3/Eigen/Eigen>
#include <vector>
#include <string>

#include "condition_functions.h"

int main(int argc, char* argv[])
{

    // argument 1: type of boundary condition for x boundaries
    // argument 2: type of boundary condition for y boundaries

    std::string xBCType = argv[1];
    std::string yBCType = argv[2];

    std::string neumannBCScheme;

    if(xBCType == "neumann" || yBCType == "neumann")
    {
        neumannBCScheme = argv[3];
    }

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

    // Solution matrix for the next timestep
    Eigen::MatrixXd nextTimestepMatrix(ySpacePoints.size(), xSpacePoints.size());
    nextTimestepMatrix.setZero();

    // IC

    for(int k = 0; k < U.rows(); k++)
    {
        for(int j = 0; j < U.cols(); j++)
        {
            U(k, j) = ic_func(xSpacePoints(j, 0), ySpacePoints(k, 0));
        }
    }

    // BCs
    // The application of Dirichlet BCs differs depending on the combination of BC types
    // and also the scheme used to solve for Neumann BCs

    if((xBCType == "dirichlet" && yBCType == "dirichlet") || (xBCType == "neumann" || yBCType == "neumann") && neumannBCScheme == "onesided")
    {

        // y boundaries
        if(yBCType == "dirichlet")
        {
            for(int j = 1; j < U.cols() - 1; j++)
            {
                U(0, j) = y_lower_dirichlet_bc_func(xSpacePoints(j, 0), timePoints(0, 0));
                U(U.rows() - 1, j) = y_upper_dirichlet_bc_func(xSpacePoints(j, 0), timePoints(0, 0));
            }
        }

        // x boundaries
        if(xBCType == "dirichlet")
        {
            for(int k = 0; k < U.rows(); k++)
            {
                U(k, 0) = x_lhs_dirichlet_bc_func(ySpacePoints(k, 0), timePoints(0, 0));
                U(k, U.cols() - 1) = x_rhs_dirichlet_bc_func(ySpacePoints(k, 0), timePoints(0, 0));
            }
        }

    }

    if(yBCType == "dirichlet" && xBCType == "neumann" && neumannBCScheme == "ghost")
    {
        // y boundaries
        for(int j = 0; j < U.cols(); j++)
        {
            U(0, j) = y_lower_dirichlet_bc_func(xSpacePoints(j, 0), timePoints(0, 0));
            U(U.rows() - 1, j) = y_upper_dirichlet_bc_func(xSpacePoints(j, 0), timePoints(0, 0));
        }
    }

    if(xBCType == "dirichlet" && yBCType == "neumann" && neumannBCScheme == "ghost")
    {
        // x boundaries
        for(int k = 0; k < U.rows(); k++)
        {
            U(k, 0) = x_lhs_dirichlet_bc_func(ySpacePoints(k, 0), timePoints(0, 0));
            U(k, U.cols() - 1) = x_rhs_dirichlet_bc_func(ySpacePoints(k, 0), timePoints(0, 0));
        }
    }

    // Define submatrices that will form the block matrices in the method equation

    int xMatricesDim, yMatricesDim;

    if(xBCType == "dirichlet" && yBCType == "dirichlet")
    {
        xMatricesDim = N_x - 1;
        yMatricesDim = N_y - 1;
    }
    else if(neumannBCScheme == "onesided")
    {
        xMatricesDim = N_x - 1;
        yMatricesDim = N_y - 1;
    }
    else if(xBCType == "neumann" && yBCType == "dirichlet" && neumannBCScheme == "ghost")
    {
        xMatricesDim = N_x + 1;
        yMatricesDim = N_y - 1;
    }
    else if(xBCType == "dirichlet" && yBCType == "neumann" && neumannBCScheme == "ghost")
    {
        xMatricesDim = N_x - 1;
        yMatricesDim = N_y + 1;
    }
    else if(xBCType == "neumann" && yBCType == "neumann" && neumannBCScheme == "ghost")
    {
        xMatricesDim = N_x + 1;
        yMatricesDim = N_y + 1;
    }

    Eigen::internal::BandMatrix<double> D_x(xMatricesDim, xMatricesDim, 1, 1);
    Eigen::internal::BandMatrix<double> D_y(yMatricesDim, yMatricesDim, 1, 1);
    Eigen::MatrixXd I_x = Eigen::MatrixXd::Identity(xMatricesDim, xMatricesDim);
    Eigen::MatrixXd I_y = Eigen::MatrixXd::Identity(yMatricesDim, yMatricesDim);

    for(int i = -D_x.subs(); i <= D_x.supers(); i++)
    {
        D_x.diagonal(i).setConstant(0);
        D_y.diagonal(i).setConstant(0);
    }

    // Define some useful constants

    double q = dt/pow(dx, 2.0);
    double r = dt/pow(dy, 2.0);

    D_x.diagonal(0).setConstant(-2.0 * q);
    D_x.diagonal(-1).setConstant(1.0 * q);
    D_x.diagonal(1).setConstant(1.0 * q);

    D_y.diagonal(0).setConstant(-2.0 * r);
    D_y.diagonal(-1).setConstant(1.0 * r);
    D_y.diagonal(1).setConstant(1.0 * r);

    if(xBCType == "neumann" && neumannBCScheme == "ghost")
    {
        D_x.diagonal(1)(0) = 2 * q;
        D_x.diagonal(-1)(xMatricesDim - 2) = 2 * q;
    }

    if(yBCType == "neumann" && neumannBCScheme == "ghost")
    {
        D_y.diagonal(1)(0) = 2 * r;
        D_y.diagonal(-1)(yMatricesDim - 2) = 2 * r;
    }

    // Define Kronecker product matrices

    Eigen::MatrixXd kroneckerProdMatrixA(I_y.rows() * D_x.rows(), I_y.cols() * D_x.cols());
    Eigen::MatrixXd kroneckerProdMatrixB(D_y.rows() * I_x.rows(), D_y.cols() * I_x.cols());

    for(int i = 0; i < I_y.rows(); i++)
    {
        for(int j = 0; j < I_y.cols(); j++)
        {
            kroneckerProdMatrixA.block(i * D_x.rows(), j * D_x.cols(), D_x.rows(), D_x.cols()) = I_y(i, j) * D_x.toDenseMatrix();
        }
    }

    for(int i = 0; i < D_y.rows(); i++)
    {
        for(int j = 0; j < D_y.cols(); j++)
        {
            kroneckerProdMatrixB.block(i * I_x.rows(), j * I_x.cols(), I_x.rows(), I_x.cols()) = D_y.toDenseMatrix()(i, j) * I_x;
        }
    }

    // Create vector that will contain the solution matrices for all timesteps

    std::vector<Eigen::MatrixXd> solutionMatrices(noOfTimePoints);
    solutionMatrices[0] = U;

    Eigen::MatrixXd b((xMatricesDim) * (yMatricesDim), 1);
    Eigen::MatrixXd previousTimestepVector((xMatricesDim) * (yMatricesDim), 1);
    Eigen::MatrixXd nextTimestepVector((xMatricesDim) * (yMatricesDim), 1);

    int xLowerBound, xUpperBound, yLowerBound, yUpperBound;

    // Inclusive lower bound
    if(xBCType == "dirichlet")
    {
        xLowerBound = 1;
    }
    else if(xBCType == "neumann")
    {
        if(neumannBCScheme == "onesided")
        {
            xLowerBound = 1;
        }
        else if(neumannBCScheme == "ghost")
        {
            xLowerBound = 0;
        }
    }

    if(yBCType == "dirichlet")
    {
        yLowerBound = 1;
    }
    else if(yBCType == "neumann")
    {
        if(neumannBCScheme == "onesided")
        {
            yLowerBound = 1;
        }
        else if(neumannBCScheme == "ghost")
        {
            yLowerBound = 0;
        }
    }

    // Exclusive upper bound
    if(xBCType == "dirichlet")
    {
        xUpperBound = N_x - 1;
    }
    else if(xBCType == "neumann")
    {
        if(neumannBCScheme == "onesided")
        {
            xUpperBound = N_x;
        }
        else if(neumannBCScheme == "ghost")
        {
            xUpperBound = N_x + 1;
        }
    }

    if(yBCType == "dirichlet")
    {
        yUpperBound = N_y - 1;
    }
    else if(yBCType == "neumann")
    {
        if(neumannBCScheme == "onesided")
        {
            yUpperBound = N_y;
        }
        else if(neumannBCScheme == "ghost")
        {
            yUpperBound = N_y + 1;
        }
    }

    // FTCS
    // PDE: u_t = u_xx + u_yy

    for(int n = 1; n < noOfTimePoints; n++)
    {

        // Set the BCs for the solution matrix of the next timestep

        if((xBCType == "dirichlet" && yBCType == "dirichlet") || (xBCType == "neumann" || yBCType == "neumann") && neumannBCScheme == "onesided")
        {

            // y boundaries
            if(yBCType == "dirichlet")
            {
                for(int j = 1; j < nextTimestepMatrix.cols() - 1; j++)
                {
                    nextTimestepMatrix(0, j) = y_lower_dirichlet_bc_func(xSpacePoints(j, 0), timePoints(n, 0));
                    nextTimestepMatrix(nextTimestepMatrix.rows() - 1, j) = y_upper_dirichlet_bc_func(xSpacePoints(j, 0), timePoints(n, 0));
                }
            }

            // x boundaries
            if(xBCType == "dirichlet")
            {
                for(int k = 0; k < nextTimestepMatrix.rows(); k++)
                {
                    nextTimestepMatrix(k, 0) = x_lhs_dirichlet_bc_func(ySpacePoints(k, 0), timePoints(n, 0));
                    nextTimestepMatrix(k, nextTimestepMatrix.cols() - 1) = x_rhs_dirichlet_bc_func(ySpacePoints(k, 0), timePoints(n, 0));
                }
            }

        }

        if(yBCType == "dirichlet" && xBCType == "neumann" && neumannBCScheme == "ghost")
        {
            // y boundaries
            for(int j = 0; j < nextTimestepMatrix.cols(); j++)
            {
                nextTimestepMatrix(0, j) = y_lower_dirichlet_bc_func(xSpacePoints(j, 0), timePoints(n, 0));
                nextTimestepMatrix(nextTimestepMatrix.rows() - 1, j) = y_upper_dirichlet_bc_func(xSpacePoints(j, 0), timePoints(n, 0));
            }
        }

        if(xBCType == "dirichlet" && yBCType == "neumann" && neumannBCScheme == "ghost")
        {
            // x boundaries
            for(int k = 0; k < nextTimestepMatrix.rows(); k++)
            {
                nextTimestepMatrix(k, 0) = x_lhs_dirichlet_bc_func(ySpacePoints(k, 0), timePoints(n, 0));
                nextTimestepMatrix(k, nextTimestepMatrix.cols() - 1) = x_rhs_dirichlet_bc_func(ySpacePoints(k, 0), timePoints(n, 0));
            }
        }

        // Grab the previous timestep's solution matrix for convenience
        Eigen::MatrixXd previousTimestepMatrix = solutionMatrices[n-1];

        b.setZero();
        previousTimestepVector.setZero();
        nextTimestepVector.setZero();

        for(int i = 0; i < yMatricesDim; i++)
        {
            for(int j = 0; j < xMatricesDim; j++)
            {
                previousTimestepVector(xMatricesDim * i + j, 0) = previousTimestepMatrix(i + yLowerBound, j + xLowerBound);
            }
        }

        // Putting the appropriate boundary points into the vector b
        if((xBCType == "dirichlet" && yBCType == "dirichlet") || (xBCType == "neumann" || yBCType == "neumann") && neumannBCScheme == "onesided")
        {
            for(int i = 1; i < previousTimestepMatrix.rows() - 1; i++)
            {
                for(int j = 1; j < previousTimestepMatrix.cols() - 1; j++)
                {
                    if(i - 1 == 0)
                    {
                        b((i-1) * (previousTimestepMatrix.cols() - 2) + (j-1), 0) += r * previousTimestepMatrix(i-1, j);
                    }
                    else if(i + 1 == N_y)
                    {
                        b((i-1) * (previousTimestepMatrix.cols() - 2) + (j-1), 0) += r * previousTimestepMatrix(i+1, j);
                    }

                    if(j - 1 == 0)
                    {
                        b((i-1) * (previousTimestepMatrix.cols() - 2) + (j-1), 0) += q * previousTimestepMatrix(i, j-1);
                    }
                    else if(j + 1 == N_x)
                    {
                        b((i-1) * (previousTimestepMatrix.cols() - 2) + (j-1), 0) += q * previousTimestepMatrix(i, j+1);
                    }
                }
            }
        }

        // If there are Neumann BCs to be solved using ghost points, need to put additional terms
        // in the vector b

        if(yBCType == "dirichlet" && xBCType == "neumann" && neumannBCScheme == "ghost")
        {
            for(int i = yLowerBound; i < yUpperBound + 1; i++)
            {
                for(int j = xLowerBound; j < xUpperBound + 1; j++)
                {
                    if(j == xLowerBound && i == yLowerBound)
                    {
                        b(0, 0) += -(dt/dx) * x_lhs_neumann_bc_func(ySpacePoints(i, 0), timePoints(n, 0));
                        b(0, 0) += q * previousTimestepMatrix(0, 0);
                    }
                    else if(j == xLowerBound && i > yLowerBound && i < yUpperBound)
                    {
                        b(xMatricesDim * (i - yLowerBound), 0) += -(dt/dx) * x_lhs_neumann_bc_func(ySpacePoints(i, 0), timePoints(n, 0));
                    }
                    else if(j == xLowerBound && i == yUpperBound)
                    {
                        b(xMatricesDim * (yMatricesDim - 1), 0) += -(dt/dx) * x_lhs_neumann_bc_func(ySpacePoints(i, 0), timePoints(n, 0));
                        b(xMatricesDim * (yMatricesDim - 1), 0) += q * previousTimestepMatrix(N_y, 0);
                    }

                    if(j == xUpperBound && i == yLowerBound)
                    {
                        b(xMatricesDim - 1, 0) += (dt/dx) * x_rhs_neumann_bc_func(ySpacePoints(i, 0), timePoints(n, 0));
                        b(xMatricesDim - 1, 0) += q * previousTimestepMatrix(0, xMatricesDim - 1);
                    }
                    else if(j == xUpperBound && i > yLowerBound && i < yUpperBound)
                    {
                        b(xMatricesDim * (i - yLowerBound) + (xMatricesDim - 1), 0) += (dt/dx) * x_rhs_neumann_bc_func(ySpacePoints(i, 0), timePoints(n, 0));
                    }
                    else if(j == xUpperBound && i == yUpperBound)
                    {
                        b(xMatricesDim * yMatricesDim - 1, 0) += (dt/dx) * x_rhs_neumann_bc_func(ySpacePoints(i, 0), timePoints(n, 0));
                        b(xMatricesDim * yMatricesDim - 1, 0) += q * previousTimestepMatrix(N_y, N_x);
                    }

                }
            }
        }

        if(xBCType == "dirichlet" && yBCType == "neumann" && neumannBCScheme == "ghost")
        {
            for(int i = yLowerBound; i < yUpperBound + 1; i++)
            {
                for(int j = xLowerBound; j < xUpperBound + 1; j++)
                {
                    if(i == yLowerBound && j == xLowerBound)
                    {
                        b(0, 0) += -(dt/dy) * y_lower_neumann_bc_func(xSpacePoints(j, 0), timePoints(n, 0));
                        b(0, 0) += r * previousTimestepMatrix(0, 0);
                    }
                    else if(i == yLowerBound && j > xLowerBound && j < xUpperBound)
                    {
                        b(j - xLowerBound, 0) += -(dt/dy) * y_lower_neumann_bc_func(xSpacePoints(j, 0), timePoints(n, 0));
                    }
                    else if(i == yLowerBound && j == xUpperBound)
                    {
                        b(j - xLowerBound, 0) += -(dt/dy) * y_lower_neumann_bc_func(xSpacePoints(j, 0), timePoints(n, 0));
                        b(j - xLowerBound, 0) += r * previousTimestepMatrix(0, N_x);
                    }

                    if(i == yUpperBound && j == xLowerBound)
                    {
                        b(xMatricesDim * (yMatricesDim - 1), 0) += (dt/dy) * y_upper_neumann_bc_func(xSpacePoints(j, 0), timePoints(n, 0));
                        b(xMatricesDim * (yMatricesDim - 1), 0) += r * previousTimestepMatrix(N_y, 0);
                    }
                    else if(i == yUpperBound && j > xLowerBound && j < xUpperBound)
                    {
                        b(xMatricesDim * (yMatricesDim - 1) + j - xLowerBound, 0) += (dt/dy) * y_upper_neumann_bc_func(xSpacePoints(j, 0), timePoints(n, 0));
                    }
                    else if(i == yUpperBound && j == xUpperBound)
                    {
                        b(xMatricesDim * yMatricesDim - 1, 0) += (dt/dy) * y_upper_neumann_bc_func(xSpacePoints(j, 0), timePoints(n, 0));
                        b(xMatricesDim * yMatricesDim - 1, 0) += r * previousTimestepMatrix(N_y, N_x);
                    }

                }
            }
        }

        if(xBCType == "neumann" && yBCType == "neumann" && neumannBCScheme == "ghost")
        {
            for(int i = yLowerBound; i < yUpperBound ; i++)
            {
                for(int j = xLowerBound; j < xUpperBound; j++)
                {
                    if(j == xLowerBound)
                    {
                        b(xMatricesDim * (i - yLowerBound), 0) += -(dt/dx) * x_lhs_neumann_bc_func(ySpacePoints(i, 0), timePoints(n, 0));
                    }
                    else if(j == xUpperBound - 1)
                    {
                        b(xMatricesDim * (i - yLowerBound) + j, 0) += (dt/dx) * x_rhs_neumann_bc_func(ySpacePoints(i, 0), timePoints(n, 0));
                    }

                    if(i == yLowerBound)
                    {
                        b(j - xLowerBound, 0) += -(dt/dy) * y_lower_neumann_bc_func(xSpacePoints(j, 0), timePoints(n, 0));
                    }
                    else if(i == yUpperBound - 1)
                    {
                        b(xMatricesDim * (yMatricesDim - 1) + j - xLowerBound, 0) += (dt/dy) * y_upper_neumann_bc_func(xSpacePoints(j, 0), timePoints(n, 0));
                    }
                }
            }
        }

        nextTimestepVector = (Eigen::MatrixXd::Identity(xMatricesDim * yMatricesDim, xMatricesDim * yMatricesDim) + kroneckerProdMatrixA + kroneckerProdMatrixB) * previousTimestepVector + b;

        // Copy info from nextTimestepVector into nextTimestepMatrix

        for(int i = 0; i < yMatricesDim; i++)
        {
            for(int j = 0; j < xMatricesDim; j++)
            {
                nextTimestepMatrix(i + yLowerBound, j + xLowerBound) = nextTimestepVector(xMatricesDim * i + j, 0);
            }
        }

        // Put nextTimestepMatrix into the vector of solution matrices

        solutionMatrices[n] = nextTimestepMatrix;

        // Reset nextTimestepMatrix in preparation for the next timestep

        nextTimestepMatrix.setZero();

        if(yBCType == "neumann" && neumannBCScheme == "onesided")
        {
            for(int j = 1; j < nextTimestepMatrix.cols() - 1; j++)
            {
                nextTimestepMatrix(0, j) = nextTimestepMatrix(1, j) - dy * y_lower_neumann_bc_func(xSpacePoints(j, 0), timePoints(n, 0));
                nextTimestepMatrix(N_y, j) = nextTimestepMatrix(N_y - 1, j) + dy * y_upper_neumann_bc_func(xSpacePoints(j, 0), timePoints(n, 0));
            }
        }

        if(xBCType == "neumann" && neumannBCScheme == "onesided")
        {
            for(int k = 0; k < nextTimestepMatrix.rows(); k++)
            {
                nextTimestepMatrix(k, 0) = nextTimestepMatrix(k, 1) - dx * x_lhs_neumann_bc_func(ySpacePoints(k, 0), timePoints(n, 0));
                nextTimestepMatrix(k, N_x) = nextTimestepMatrix(k, N_x - 1) + dx * x_rhs_neumann_bc_func(ySpacePoints(k, 0), timePoints(n, 0));
            }
        }

    }

}
