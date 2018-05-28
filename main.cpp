#include <iostream>
#include <eigen3/Eigen/Eigen>
#include <vector>

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

    // Solution matrix for the next timestep
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
        U(k, 0) = x_lhs_dirichlet_bc_func(ySpacePoints(k, 0));
        U(k, U.cols() - 1) = x_rhs_dirichlet_bc_func(ySpacePoints(k, 0));
    }

    // y boundaries
    for(int j = 0; j < U.cols(); j++)
    {
        U(0, j) = y_lower_dirichlet_bc_func(xSpacePoints(j, 0));
        U(U.rows() - 1, j) = y_upper_dirichlet_bc_func(xSpacePoints(j, 0));
    }

    // Define submatrices that will form the block matrices in the method equation

    Eigen::internal::BandMatrix<double> D_x(N_x - 1, N_x - 1, 1, 1);
    Eigen::internal::BandMatrix<double> D_y(N_y - 1, N_y - 1, 1, 1);
    Eigen::MatrixXd I_x = Eigen::MatrixXd::Identity(N_x - 1, N_x - 1);
    Eigen::MatrixXd I_y = Eigen::MatrixXd::Identity(N_y - 1, N_y - 1);

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

    Eigen::MatrixXd b((N_x - 1) * (N_y - 1), 1);
    Eigen::MatrixXd previousTimestepVector((N_x - 1) * (N_y - 1), 1);
    Eigen::MatrixXd nextTimestep((N_x - 1) * (N_y - 1), 1);

    // FTCS
    // PDE: u_t = u_xx + u_yy

    for(int n = 1; n < noOfTimePoints; n++)
    {

        // Set the BCs for the solution matrix of the next timestep

        // x boundaries
        for(int k = 0; k < V.rows(); k++)
        {
            V(k, 0) = x_lhs_dirichlet_bc_func(ySpacePoints(k, 0));
            V(k, V.cols() - 1) = x_rhs_dirichlet_bc_func(ySpacePoints(k, 0));
        }

        // y boundaries
        for(int j = 0; j < V.cols(); j++)
        {
            V(0, j) = y_lower_dirichlet_bc_func(xSpacePoints(j, 0));
            V(V.rows() - 1, j) = y_upper_dirichlet_bc_func(xSpacePoints(j, 0));
        }

        // Grab the previous timestep's solution matrix for convenience
        Eigen::MatrixXd previousTimestepMatrix = solutionMatrices[n-1];

        b.setZero();
        previousTimestepVector.setZero();
        nextTimestep.setZero();

        for(int i = 1; i < previousTimestepMatrix.rows() - 1; i++)
        {
            for(int j = 1; j < previousTimestepMatrix.cols() - 1; j++)
            {
                previousTimestepVector((i-1) * (previousTimestepMatrix.cols() - 2) + (j-1), 0) = previousTimestepMatrix(i, j);
            }
        }

        // Putting the appropriate boundary points into the vector b

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

        nextTimestep = (Eigen::MatrixXd::Identity((N_x - 1) * (N_y - 1), (N_x - 1) * (N_y - 1)) + kroneckerProdMatrixA + kroneckerProdMatrixB) * previousTimestepVector + b;

        // Copy info from nextTimestep into V

        for(int i = 1; i < previousTimestepMatrix.rows() - 1; i++)
        {
            for(int j = 1; j < previousTimestepMatrix.cols() - 1; j++)
            {
                V(i, j) = nextTimestep((i-1) * (V.cols() - 2) + (j-1), 0);
            }
        }

        // Put V into the vector of solution matrices

        solutionMatrices[n] = V;

        // Reset V in preparation for the next timestep

        V.setZero();

    }

}
