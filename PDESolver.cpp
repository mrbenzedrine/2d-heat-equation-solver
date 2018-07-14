#include "PDESolver.h"

PDESolver::PDESolver(
        int N_x,
        int N_y,
        double dt,
        double startX,
        double endX,
        double startY,
        double endY,
        double startT,
        std::string xBCType,
        std::string yBCType,
        PDEConditionFunctions conditionFuncs,
        double theta
):N_x(N_x), N_y(N_y), dt(dt),
    startX(startX), endX(endX),
    startY(startY), endY(endY),
    startT(startT),
    xBCType(xBCType), yBCType(yBCType),
    conditionFuncs(conditionFuncs),
    theta(theta)
{
    initialise_vectors_matrices();
    perform_prelim_mathematical_routines();
}

PDESolver::PDESolver(
        int N_x,
        int N_y,
        double dt,
        double startX,
        double endX,
        double startY,
        double endY,
        double startT,
        std::string xBCType,
        std::string yBCType,
        std::string neumannBCScheme,
        PDEConditionFunctions conditionFuncs,
        double theta
):N_x(N_x), N_y(N_y), dt(dt),
    startX(startX), endX(endX),
    startY(startY), endY(endY),
    startT(startT),
    xBCType(xBCType), yBCType(yBCType),
    neumannBCScheme(neumannBCScheme),
    conditionFuncs(conditionFuncs),
    theta(theta)
{
    initialise_vectors_matrices();
    perform_prelim_mathematical_routines();
}

void PDESolver::initialise_vectors_matrices()
{
    create_discretisation_vectors();
    create_solution_matrix();
}

void PDESolver::create_discretisation_vectors()
{
    xSpacePoints = Eigen::VectorXd::LinSpaced(N_x + 1, startX, endX);
    ySpacePoints = Eigen::VectorXd::LinSpaced(N_y + 1, startY, endY);
}

void PDESolver::create_solution_matrix()
{
    nextTimestepMatrix = Eigen::MatrixXd(ySpacePoints.size(), xSpacePoints.size());
}

void PDESolver::perform_prelim_mathematical_routines()
{
    apply_prelim_ic_bcs();
    create_kronecker_product_matrices();
    get_solution_looping_bounds();
}

void PDESolver::apply_prelim_ic_bcs()
{
    // IC
    for(int k = 0; k < nextTimestepMatrix.rows(); k++)
    {
        for(int j = 0; j < nextTimestepMatrix.cols(); j++)
        {
            nextTimestepMatrix(k, j) = conditionFuncs.ic_func(xSpacePoints(j, 0), ySpacePoints(k, 0));
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
            for(int j = 1; j < nextTimestepMatrix.cols() - 1; j++)
            {
                nextTimestepMatrix(0, j) = conditionFuncs.y_lower_dirichlet_bc_func(xSpacePoints(j, 0), startT);
                nextTimestepMatrix(nextTimestepMatrix.rows() - 1, j) = conditionFuncs.y_upper_dirichlet_bc_func(xSpacePoints(j, 0), startT);
            }
        }

        // x boundaries
        if(xBCType == "dirichlet")
        {
            for(int k = 0; k < nextTimestepMatrix.rows(); k++)
            {
                nextTimestepMatrix(k, 0) = conditionFuncs.x_lhs_dirichlet_bc_func(ySpacePoints(k, 0), startT);
                nextTimestepMatrix(k, nextTimestepMatrix.cols() - 1) = conditionFuncs.x_rhs_dirichlet_bc_func(ySpacePoints(k, 0), startT);
            }
        }

    }

    if(yBCType == "dirichlet" && xBCType == "neumann" && neumannBCScheme == "ghost")
    {
        // y boundaries
        for(int j = 0; j < nextTimestepMatrix.cols(); j++)
        {
            nextTimestepMatrix(0, j) = conditionFuncs.y_lower_dirichlet_bc_func(xSpacePoints(j, 0), startT);
            nextTimestepMatrix(nextTimestepMatrix.rows() - 1, j) = conditionFuncs.y_upper_dirichlet_bc_func(xSpacePoints(j, 0), startT);
        }
    }

    if(xBCType == "dirichlet" && yBCType == "neumann" && neumannBCScheme == "ghost")
    {
        // x boundaries
        for(int k = 0; k < nextTimestepMatrix.rows(); k++)
        {
            nextTimestepMatrix(k, 0) = conditionFuncs.x_lhs_dirichlet_bc_func(ySpacePoints(k, 0), startT);
            nextTimestepMatrix(k, nextTimestepMatrix.cols() - 1) = conditionFuncs.x_rhs_dirichlet_bc_func(ySpacePoints(k, 0), startT);
        }
    }

}

void PDESolver::create_kronecker_product_matrices()
{
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

    Eigen::internal::BandMatrix<double> previousTimestepD_x(xMatricesDim, xMatricesDim, 1, 1);
    Eigen::internal::BandMatrix<double> previousTimestepD_y(yMatricesDim, yMatricesDim, 1, 1);
    Eigen::internal::BandMatrix<double> nextTimestepD_x(xMatricesDim, xMatricesDim, 1, 1);
    Eigen::internal::BandMatrix<double> nextTimestepD_y(yMatricesDim, yMatricesDim, 1, 1);
    Eigen::MatrixXd I_x = Eigen::MatrixXd::Identity(xMatricesDim, xMatricesDim);
    Eigen::MatrixXd I_y = Eigen::MatrixXd::Identity(yMatricesDim, yMatricesDim);

    for(int i = -previousTimestepD_x.subs(); i <= previousTimestepD_x.supers(); i++)
    {
        previousTimestepD_x.diagonal(i).setConstant(0);
        previousTimestepD_y.diagonal(i).setConstant(0);
        nextTimestepD_x.diagonal(i).setConstant(0);
        nextTimestepD_y.diagonal(i).setConstant(0);
    }

    // Define some useful constants

    double dx = 1.0/N_x;
    double dy = 1.0/N_y;
    double q = dt/pow(dx, 2.0);
    double r = dt/pow(dy, 2.0);
    double previousTimestepKronProdMatrixAFactor = (1.0 - theta) * q;
    double previousTimestepKronProdMatrixBFactor = (1.0 - theta) * r;
    double nextTimestepKronProdMatrixAFactor = -theta * q;
    double nextTimestepKronProdMatrixBFactor = -theta * r;

    previousTimestepD_x.diagonal(0).setConstant(-2.0 * previousTimestepKronProdMatrixAFactor);
    previousTimestepD_x.diagonal(-1).setConstant(1.0 * previousTimestepKronProdMatrixAFactor);
    previousTimestepD_x.diagonal(1).setConstant(1.0 * previousTimestepKronProdMatrixAFactor);

    previousTimestepD_y.diagonal(0).setConstant(-2.0 * previousTimestepKronProdMatrixBFactor);
    previousTimestepD_y.diagonal(-1).setConstant(1.0 * previousTimestepKronProdMatrixBFactor);
    previousTimestepD_y.diagonal(1).setConstant(1.0 * previousTimestepKronProdMatrixBFactor);

    nextTimestepD_x.diagonal(0).setConstant(-2.0 * nextTimestepKronProdMatrixAFactor);
    nextTimestepD_x.diagonal(-1).setConstant(1.0 * nextTimestepKronProdMatrixAFactor);
    nextTimestepD_x.diagonal(1).setConstant(1.0 * nextTimestepKronProdMatrixAFactor);

    nextTimestepD_y.diagonal(0).setConstant(-2.0 * nextTimestepKronProdMatrixBFactor);
    nextTimestepD_y.diagonal(-1).setConstant(1.0 * nextTimestepKronProdMatrixBFactor);
    nextTimestepD_y.diagonal(1).setConstant(1.0 * nextTimestepKronProdMatrixBFactor);

    if(xBCType == "neumann" && neumannBCScheme == "onesided")
    {
        nextTimestepD_x.diagonal(0)(0) = -1 * nextTimestepKronProdMatrixAFactor;
        nextTimestepD_x.diagonal(0)(xMatricesDim - 1) = -1 * nextTimestepKronProdMatrixAFactor;
    }

    if(yBCType == "neumann" && neumannBCScheme == "onesided")
    {
        nextTimestepD_y.diagonal(0)(0) = -1 * nextTimestepKronProdMatrixBFactor;
        nextTimestepD_y.diagonal(0)(yMatricesDim - 1) = -1 * nextTimestepKronProdMatrixBFactor;
    }

    if(xBCType == "neumann" && neumannBCScheme == "ghost")
    {
        nextTimestepD_x.diagonal(1)(0) = 2 * nextTimestepKronProdMatrixAFactor;
        nextTimestepD_x.diagonal(-1)(xMatricesDim - 2) = 2 * nextTimestepKronProdMatrixAFactor;
    }

    if(yBCType == "neumann" && neumannBCScheme == "ghost")
    {
        nextTimestepD_y.diagonal(1)(0) = 2 * nextTimestepKronProdMatrixBFactor;
        nextTimestepD_y.diagonal(-1)(yMatricesDim - 2) = 2 * nextTimestepKronProdMatrixBFactor;
    }

    // Define Kronecker product matrices

    previousTimestepKronProdMatrixA = kronecker_product(I_y, previousTimestepD_x.toDenseMatrix());
    previousTimestepKronProdMatrixB = kronecker_product(previousTimestepD_y.toDenseMatrix(), I_x);
    nextTimestepKronProdMatrixA = kronecker_product(I_y, nextTimestepD_x.toDenseMatrix());
    nextTimestepKronProdMatrixB = kronecker_product(nextTimestepD_y.toDenseMatrix(), I_x);
}

Eigen::MatrixXd PDESolver::kronecker_product(Eigen::MatrixXd A, Eigen::MatrixXd B)
{
    Eigen::MatrixXd kroneckerProduct(A.rows() * B.rows(), A.cols() * B.cols());

    for(int i = 0; i < A.rows(); i++)
    {
        for(int j = 0; j < A.cols(); j++)
        {
            kroneckerProduct.block(i * B.rows(), j * B.cols(), B.rows(), B.cols()) = A(i, j) * B;
        }
    }

    return kroneckerProduct;
}

void PDESolver::get_solution_looping_bounds()
{
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
}

void PDESolver::solve_next_timestep(int timestep_no)
{
    Eigen::MatrixXd b((xMatricesDim) * (yMatricesDim), 1);
    Eigen::MatrixXd previousTimestepVector((xMatricesDim) * (yMatricesDim), 1);
    Eigen::MatrixXd nextTimestepVector((xMatricesDim) * (yMatricesDim), 1);

    // Grab the previous timestep's solution matrix
    Eigen::MatrixXd previousTimestepMatrix = nextTimestepMatrix;

    // Set the BCs for the solution matrix of the next timestep

    if((xBCType == "dirichlet" && yBCType == "dirichlet") || (xBCType == "neumann" || yBCType == "neumann") && neumannBCScheme == "onesided")
    {

        // y boundaries
        if(yBCType == "dirichlet")
        {
            for(int j = 1; j < nextTimestepMatrix.cols() - 1; j++)
            {
                nextTimestepMatrix(0, j) = conditionFuncs.y_lower_dirichlet_bc_func(xSpacePoints(j, 0), timestep_no * dt);
                nextTimestepMatrix(nextTimestepMatrix.rows() - 1, j) = conditionFuncs.y_upper_dirichlet_bc_func(xSpacePoints(j, 0), timestep_no * dt);
            }
        }

        // x boundaries
        if(xBCType == "dirichlet")
        {
            for(int k = 0; k < nextTimestepMatrix.rows(); k++)
            {
                nextTimestepMatrix(k, 0) = conditionFuncs.x_lhs_dirichlet_bc_func(ySpacePoints(k, 0), timestep_no * dt);
                nextTimestepMatrix(k, nextTimestepMatrix.cols() - 1) = conditionFuncs.x_rhs_dirichlet_bc_func(ySpacePoints(k, 0), timestep_no * dt);
            }
        }

    }

    if(yBCType == "dirichlet" && xBCType == "neumann" && neumannBCScheme == "ghost")
    {
        // y boundaries
        for(int j = 0; j < nextTimestepMatrix.cols(); j++)
        {
            nextTimestepMatrix(0, j) = conditionFuncs.y_lower_dirichlet_bc_func(xSpacePoints(j, 0), timestep_no * dt);
            nextTimestepMatrix(nextTimestepMatrix.rows() - 1, j) = conditionFuncs.y_upper_dirichlet_bc_func(xSpacePoints(j, 0), timestep_no * dt);
        }
    }

    if(xBCType == "dirichlet" && yBCType == "neumann" && neumannBCScheme == "ghost")
    {
        // x boundaries
        for(int k = 0; k < nextTimestepMatrix.rows(); k++)
        {
            nextTimestepMatrix(k, 0) = conditionFuncs.x_lhs_dirichlet_bc_func(ySpacePoints(k, 0), timestep_no * dt);
            nextTimestepMatrix(k, nextTimestepMatrix.cols() - 1) = conditionFuncs.x_rhs_dirichlet_bc_func(ySpacePoints(k, 0), timestep_no * dt);
        }
    }

    b.setZero();

    for(int i = 0; i < yMatricesDim; i++)
    {
        for(int j = 0; j < xMatricesDim; j++)
        {
            previousTimestepVector(xMatricesDim * i + j, 0) = previousTimestepMatrix(i + yLowerBound, j + xLowerBound);
        }
    }

    double dx = 1.0/N_x;
    double dy = 1.0/N_y;
    double q = dt/pow(dx, 2.0);
    double r = dt/pow(dy, 2.0);
    double previousTimestepMatrixAFactor = (1.0 - theta) * q;
    double previousTimestepMatrixBFactor = (1.0 - theta) * r;
    double nextTimestepMatrixAFactor = -theta * q;
    double nextTimestepMatrixBFactor = -theta * r;

    // Putting the appropriate boundary points into the vector b
    if((xBCType == "dirichlet" && yBCType == "dirichlet") || (xBCType == "neumann" || yBCType == "neumann") && neumannBCScheme == "onesided")
    {
        for(int i = 1; i < previousTimestepMatrix.rows() - 1; i++)
        {
            for(int j = 1; j < previousTimestepMatrix.cols() - 1; j++)
            {
                if(i - 1 == 0)
                {
                    b((i-1) * (previousTimestepMatrix.cols() - 2) + (j-1), 0) += previousTimestepMatrixBFactor * previousTimestepMatrix(i-1, j);
                }
                else if(i + 1 == N_y)
                {
                    b((i-1) * (previousTimestepMatrix.cols() - 2) + (j-1), 0) += previousTimestepMatrixBFactor * previousTimestepMatrix(i+1, j);
                }

                if(j - 1 == 0)
                {
                    b((i-1) * (previousTimestepMatrix.cols() - 2) + (j-1), 0) += previousTimestepMatrixAFactor * previousTimestepMatrix(i, j-1);
                }
                else if(j + 1 == N_x)
                {
                    b((i-1) * (previousTimestepMatrix.cols() - 2) + (j-1), 0) += previousTimestepMatrixAFactor * previousTimestepMatrix(i, j+1);
                }
            }
        }
    }

    // If there are Neumann BCs to be solved using a onesided scheme, need to put additional
    // terms in the vector b

    if(yBCType == "neumann" && neumannBCScheme == "onesided")
    {
        for(int i = yLowerBound; i < yUpperBound + 1; i++)
        {
            for(int j = xLowerBound; j < xUpperBound + 1; j++)
            {
                if(i == yLowerBound && j == xLowerBound)
                {
                    b(0, 0) += -dy * nextTimestepMatrixBFactor * conditionFuncs.y_lower_neumann_bc_func(xSpacePoints(j, 0), timestep_no * dt);
                }
                else if(i == yLowerBound && j > xLowerBound && j < xUpperBound)
                {
                    b(j - xLowerBound, 0) += -dy * nextTimestepMatrixBFactor * conditionFuncs.y_lower_neumann_bc_func(xSpacePoints(j, 0), timestep_no * dt);
                }
                else if(i == yLowerBound && j == xUpperBound)
                {
                    b(j - xLowerBound, 0) += -dy * nextTimestepMatrixBFactor * conditionFuncs.y_lower_neumann_bc_func(xSpacePoints(j, 0), timestep_no * dt);
                }

                if(i == yUpperBound && j == xLowerBound)
                {
                    b(xMatricesDim * (yMatricesDim - 1), 0) += dy * nextTimestepMatrixBFactor * conditionFuncs.y_upper_neumann_bc_func(xSpacePoints(j, 0), timestep_no * dt);
                }
                else if(i == yUpperBound && j > xLowerBound && j < xUpperBound)
                {
                    b(xMatricesDim * (yMatricesDim - 1) + j - xLowerBound, 0) += dy * nextTimestepMatrixBFactor * conditionFuncs.y_upper_neumann_bc_func(xSpacePoints(j, 0), timestep_no * dt);
                }
                else if(i == yUpperBound && j == xUpperBound)
                {
                    b(xMatricesDim * yMatricesDim - 1, 0) += dy * nextTimestepMatrixBFactor * conditionFuncs.y_upper_neumann_bc_func(xSpacePoints(j, 0), timestep_no * dt);
                }

            }
        }
    }

    if(xBCType == "neumann" && neumannBCScheme == "onesided")
    {
        for(int i = yLowerBound; i < yUpperBound + 1; i++)
        {
            for(int j = xLowerBound; j < xUpperBound + 1; j++)
            {
                if(j == xLowerBound && i == yLowerBound)
                {
                    b(0, 0) += -dx * nextTimestepMatrixAFactor * conditionFuncs.x_lhs_neumann_bc_func(ySpacePoints(i, 0), timestep_no * dt);
                }
                else if(j == j == xLowerBound && i > yLowerBound && i < yUpperBound)
                {
                    b(xMatricesDim * (i - yLowerBound), 0) += -dx * nextTimestepMatrixAFactor * conditionFuncs.x_lhs_neumann_bc_func(ySpacePoints(i, 0), timestep_no * dt);
                }
                else if(j == xLowerBound && i == yUpperBound)
                {
                    b(xMatricesDim * (yMatricesDim - 1), 0) += -dx * nextTimestepMatrixAFactor * conditionFuncs.x_lhs_neumann_bc_func(ySpacePoints(i, 0), timestep_no * dt);
                }

                if(j == xUpperBound && i == yLowerBound)
                {
                    b(xMatricesDim - 1, 0) += dx * nextTimestepMatrixAFactor * conditionFuncs.x_rhs_neumann_bc_func(ySpacePoints(i, 0), timestep_no * dt);
                }
                else if(j == xUpperBound && i > yLowerBound && i < yUpperBound)
                {
                    b(xMatricesDim * (i - yLowerBound) + (xMatricesDim - 1), 0) += dx * nextTimestepMatrixAFactor * conditionFuncs.x_rhs_neumann_bc_func(ySpacePoints(i, 0), timestep_no * dt);
                }
                else if(j == xUpperBound && i == yUpperBound)
                {
                    b(xMatricesDim * yMatricesDim - 1, 0) += dx * nextTimestepMatrixAFactor * conditionFuncs.x_rhs_neumann_bc_func(ySpacePoints(i, 0), timestep_no * dt);
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
                    b(0, 0) += -2 * dx * nextTimestepMatrixAFactor * conditionFuncs.x_lhs_neumann_bc_func(ySpacePoints(i, 0), timestep_no * dt);
                    b(0, 0) += previousTimestepMatrixAFactor * previousTimestepMatrix(0, 0);
                }
                else if(j == xLowerBound && i > yLowerBound && i < yUpperBound)
                {
                    b(xMatricesDim * (i - yLowerBound), 0) += -2 * dx * nextTimestepMatrixAFactor * conditionFuncs.x_lhs_neumann_bc_func(ySpacePoints(i, 0), timestep_no * dt);
                }
                else if(j == xLowerBound && i == yUpperBound)
                {
                    b(xMatricesDim * (yMatricesDim - 1), 0) += -2 * dx * nextTimestepMatrixAFactor * conditionFuncs.x_lhs_neumann_bc_func(ySpacePoints(i, 0), timestep_no * dt);
                    b(xMatricesDim * (yMatricesDim - 1), 0) += previousTimestepMatrixAFactor * previousTimestepMatrix(N_y, 0);
                }

                if(j == xUpperBound && i == yLowerBound)
                {
                    b(xMatricesDim - 1, 0) += 2 * dx * nextTimestepMatrixAFactor * conditionFuncs.x_rhs_neumann_bc_func(ySpacePoints(i, 0), timestep_no * dt);
                    b(xMatricesDim - 1, 0) += previousTimestepMatrixAFactor * previousTimestepMatrix(0, xMatricesDim - 1);
                }
                else if(j == xUpperBound && i > yLowerBound && i < yUpperBound)
                {
                    b(xMatricesDim * (i - yLowerBound) + (xMatricesDim - 1), 0) += 2 * dx * nextTimestepMatrixAFactor * conditionFuncs.x_rhs_neumann_bc_func(ySpacePoints(i, 0), timestep_no * dt);
                }
                else if(j == xUpperBound && i == yUpperBound)
                {
                    b(xMatricesDim * yMatricesDim - 1, 0) += 2 * dx * nextTimestepMatrixAFactor * conditionFuncs.x_rhs_neumann_bc_func(ySpacePoints(i, 0), timestep_no * dt);
                    b(xMatricesDim * yMatricesDim - 1, 0) += previousTimestepMatrixAFactor * previousTimestepMatrix(N_y, N_x);
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
                    b(0, 0) += -2 * dy * nextTimestepMatrixBFactor * conditionFuncs.y_lower_neumann_bc_func(xSpacePoints(j, 0), timestep_no * dt);
                    b(0, 0) += previousTimestepMatrixBFactor * previousTimestepMatrix(0, 0);
                }
                else if(i == yLowerBound && j > xLowerBound && j < xUpperBound)
                {
                    b(j - xLowerBound, 0) += -2 * dy * nextTimestepMatrixBFactor * conditionFuncs.y_lower_neumann_bc_func(xSpacePoints(j, 0), timestep_no * dt);
                }
                else if(i == yLowerBound && j == xUpperBound)
                {
                    b(j - xLowerBound, 0) += -2 * dy * nextTimestepMatrixBFactor * conditionFuncs.y_lower_neumann_bc_func(xSpacePoints(j, 0), timestep_no * dt);
                    b(j - xLowerBound, 0) += previousTimestepMatrixBFactor * previousTimestepMatrix(0, N_x);
                }

                if(i == yUpperBound && j == xLowerBound)
                {
                    b(xMatricesDim * (yMatricesDim - 1), 0) += 2 * dy * nextTimestepMatrixBFactor * conditionFuncs.y_upper_neumann_bc_func(xSpacePoints(j, 0), timestep_no * dt);
                    b(xMatricesDim * (yMatricesDim - 1), 0) += previousTimestepMatrixBFactor * previousTimestepMatrix(N_y, 0);
                }
                else if(i == yUpperBound && j > xLowerBound && j < xUpperBound)
                {
                    b(xMatricesDim * (yMatricesDim - 1) + j - xLowerBound, 0) += 2 * dy * nextTimestepMatrixBFactor * conditionFuncs.y_upper_neumann_bc_func(xSpacePoints(j, 0), timestep_no * dt);
                }
                else if(i == yUpperBound && j == xUpperBound)
                {
                    b(xMatricesDim * yMatricesDim - 1, 0) += 2 * dy * nextTimestepMatrixBFactor * conditionFuncs.y_upper_neumann_bc_func(xSpacePoints(j, 0), timestep_no * dt);
                    b(xMatricesDim * yMatricesDim - 1, 0) += previousTimestepMatrixBFactor * previousTimestepMatrix(N_y, N_x);
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
                    b(xMatricesDim * (i - yLowerBound), 0) += -2 * dx * nextTimestepMatrixAFactor * conditionFuncs.x_lhs_neumann_bc_func(ySpacePoints(i, 0), timestep_no * dt);
                }
                else if(j == xUpperBound - 1)
                {
                    b(xMatricesDim * (i - yLowerBound) + j, 0) += 2 * dx * nextTimestepMatrixAFactor * conditionFuncs.x_rhs_neumann_bc_func(ySpacePoints(i, 0), timestep_no * dt);
                }

                if(i == yLowerBound)
                {
                    b(j - xLowerBound, 0) += -2 * dy * nextTimestepMatrixBFactor * conditionFuncs.y_lower_neumann_bc_func(xSpacePoints(j, 0), timestep_no * dt);
                }
                else if(i == yUpperBound - 1)
                {
                    b(xMatricesDim * (yMatricesDim - 1) + j - xLowerBound, 0) += 2 * dy * nextTimestepMatrixBFactor * conditionFuncs.y_upper_neumann_bc_func(xSpacePoints(j, 0), timestep_no * dt);
                }
            }
        }
    }

    Eigen::MatrixXd idMatrix = Eigen::MatrixXd::Identity(xMatricesDim * yMatricesDim, xMatricesDim * yMatricesDim);
    Eigen::MatrixXd previousTimestepMatrices = idMatrix + previousTimestepKronProdMatrixA + previousTimestepKronProdMatrixB;
    Eigen::MatrixXd nextTimestepMatrices = idMatrix + nextTimestepKronProdMatrixA + nextTimestepKronProdMatrixB;
    nextTimestepVector = nextTimestepMatrices.inverse() * (previousTimestepMatrices * previousTimestepVector + b);

    // Copy info from nextTimestepVector into nextTimestepMatrix

    for(int i = 0; i < yMatricesDim; i++)
    {
        for(int j = 0; j < xMatricesDim; j++)
        {
            nextTimestepMatrix(i + yLowerBound, j + xLowerBound) = nextTimestepVector(xMatricesDim * i + j, 0);
        }
    }

    if(yBCType == "neumann" && neumannBCScheme == "onesided")
    {
        for(int j = 1; j < nextTimestepMatrix.cols() - 1; j++)
        {
            nextTimestepMatrix(0, j) = nextTimestepMatrix(1, j) - dy * conditionFuncs.y_lower_neumann_bc_func(xSpacePoints(j, 0), timestep_no * dt);
            nextTimestepMatrix(N_y, j) = nextTimestepMatrix(N_y - 1, j) + dy * conditionFuncs.y_upper_neumann_bc_func(xSpacePoints(j, 0), timestep_no * dt);
        }
    }

    if(xBCType == "neumann" && neumannBCScheme == "onesided")
    {
        for(int k = 0; k < nextTimestepMatrix.rows(); k++)
        {
            nextTimestepMatrix(k, 0) = nextTimestepMatrix(k, 1) - dx * conditionFuncs.x_lhs_neumann_bc_func(ySpacePoints(k, 0), timestep_no * dt);
            nextTimestepMatrix(k, N_x) = nextTimestepMatrix(k, N_x - 1) + dx * conditionFuncs.x_rhs_neumann_bc_func(ySpacePoints(k, 0), timestep_no * dt);
        }
    }

}
