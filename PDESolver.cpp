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
        double endT,
        std::string xBCType,
        std::string yBCType,
        PDEConditionFunctions conditionFuncs
):N_x(N_x), N_y(N_y), dt(dt),
    startX(startX), endX(endX),
    startY(startY), endY(endY),
    startT(startT), endT(endT),
    xBCType(xBCType), yBCType(yBCType),
    conditionFuncs(conditionFuncs)
{
    initialise_vectors_matrices();
    perform_mathematical_routines();
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
        double endT,
        std::string xBCType,
        std::string yBCType,
        std::string neumannBCScheme,
        PDEConditionFunctions conditionFuncs
):N_x(N_x), N_y(N_y), dt(dt),
    startX(startX), endX(endX),
    startY(startY), endY(endY),
    startT(startT), endT(endT),
    xBCType(xBCType), yBCType(yBCType),
    neumannBCScheme(neumannBCScheme),
    conditionFuncs(conditionFuncs)
{
    initialise_vectors_matrices();
    perform_mathematical_routines();
}

void PDESolver::initialise_vectors_matrices()
{
    create_discretisation_vectors();
    create_solution_matrices();
}

void PDESolver::create_discretisation_vectors()
{
    xSpacePoints = Eigen::VectorXd::LinSpaced(N_x + 1, startX, endX);
    ySpacePoints = Eigen::VectorXd::LinSpaced(N_y + 1, startY, endY);

    int noOfTimePoints = 0;
    double timeCounter = 0.0;

    while(timeCounter <= (endT - startT))
    {
        timeCounter += dt;
        noOfTimePoints++;
    }

    timePoints = Eigen::VectorXd::LinSpaced(noOfTimePoints, startT, endT);
}

void PDESolver::create_solution_matrices()
{
    U = Eigen::MatrixXd(ySpacePoints.size(), xSpacePoints.size());
    U.setZero();
    nextTimestepMatrix = Eigen::MatrixXd(ySpacePoints.size(), xSpacePoints.size());
    nextTimestepMatrix.setZero();
    solutionMatrices = std::vector<Eigen::MatrixXd>(timePoints.size());
}

void PDESolver::perform_mathematical_routines()
{
    apply_prelim_ic_bcs();
    create_kronecker_product_matrices();
    get_solution_looping_bounds();
    solve_pde();
}

void PDESolver::apply_prelim_ic_bcs()
{
    // IC
    for(int k = 0; k < U.rows(); k++)
    {
        for(int j = 0; j < U.cols(); j++)
        {
            U(k, j) = conditionFuncs.ic_func(xSpacePoints(j, 0), ySpacePoints(k, 0));
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
                U(0, j) = conditionFuncs.y_lower_dirichlet_bc_func(xSpacePoints(j, 0), timePoints(0, 0));
                U(U.rows() - 1, j) = conditionFuncs.y_upper_dirichlet_bc_func(xSpacePoints(j, 0), timePoints(0, 0));
            }
        }

        // x boundaries
        if(xBCType == "dirichlet")
        {
            for(int k = 0; k < U.rows(); k++)
            {
                U(k, 0) = conditionFuncs.x_lhs_dirichlet_bc_func(ySpacePoints(k, 0), timePoints(0, 0));
                U(k, U.cols() - 1) = conditionFuncs.x_rhs_dirichlet_bc_func(ySpacePoints(k, 0), timePoints(0, 0));
            }
        }

    }

    if(yBCType == "dirichlet" && xBCType == "neumann" && neumannBCScheme == "ghost")
    {
        // y boundaries
        for(int j = 0; j < U.cols(); j++)
        {
            U(0, j) = conditionFuncs.y_lower_dirichlet_bc_func(xSpacePoints(j, 0), timePoints(0, 0));
            U(U.rows() - 1, j) = conditionFuncs.y_upper_dirichlet_bc_func(xSpacePoints(j, 0), timePoints(0, 0));
        }
    }

    if(xBCType == "dirichlet" && yBCType == "neumann" && neumannBCScheme == "ghost")
    {
        // x boundaries
        for(int k = 0; k < U.rows(); k++)
        {
            U(k, 0) = conditionFuncs.x_lhs_dirichlet_bc_func(ySpacePoints(k, 0), timePoints(0, 0));
            U(k, U.cols() - 1) = conditionFuncs.x_rhs_dirichlet_bc_func(ySpacePoints(k, 0), timePoints(0, 0));
        }
    }

    solutionMatrices[0] = U;
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

    double dx = 1.0/N_x;
    double dy = 1.0/N_y;
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

    kroneckerProdMatrixA = kronecker_product(I_y, D_x.toDenseMatrix());
    kroneckerProdMatrixB = kronecker_product(D_y.toDenseMatrix(), I_x);
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

    // Set the BCs for the solution matrix of the next timestep

    if((xBCType == "dirichlet" && yBCType == "dirichlet") || (xBCType == "neumann" || yBCType == "neumann") && neumannBCScheme == "onesided")
    {

        // y boundaries
        if(yBCType == "dirichlet")
        {
            for(int j = 1; j < nextTimestepMatrix.cols() - 1; j++)
            {
                nextTimestepMatrix(0, j) = conditionFuncs.y_lower_dirichlet_bc_func(xSpacePoints(j, 0), timePoints(timestep_no, 0));
                nextTimestepMatrix(nextTimestepMatrix.rows() - 1, j) = conditionFuncs.y_upper_dirichlet_bc_func(xSpacePoints(j, 0), timePoints(timestep_no, 0));
            }
        }

        // x boundaries
        if(xBCType == "dirichlet")
        {
            for(int k = 0; k < nextTimestepMatrix.rows(); k++)
            {
                nextTimestepMatrix(k, 0) = conditionFuncs.x_lhs_dirichlet_bc_func(ySpacePoints(k, 0), timePoints(timestep_no, 0));
                nextTimestepMatrix(k, nextTimestepMatrix.cols() - 1) = conditionFuncs.x_rhs_dirichlet_bc_func(ySpacePoints(k, 0), timePoints(timestep_no, 0));
            }
        }

    }

    if(yBCType == "dirichlet" && xBCType == "neumann" && neumannBCScheme == "ghost")
    {
        // y boundaries
        for(int j = 0; j < nextTimestepMatrix.cols(); j++)
        {
            nextTimestepMatrix(0, j) = conditionFuncs.y_lower_dirichlet_bc_func(xSpacePoints(j, 0), timePoints(timestep_no, 0));
            nextTimestepMatrix(nextTimestepMatrix.rows() - 1, j) = conditionFuncs.y_upper_dirichlet_bc_func(xSpacePoints(j, 0), timePoints(timestep_no, 0));
        }
    }

    if(xBCType == "dirichlet" && yBCType == "neumann" && neumannBCScheme == "ghost")
    {
        // x boundaries
        for(int k = 0; k < nextTimestepMatrix.rows(); k++)
        {
            nextTimestepMatrix(k, 0) = conditionFuncs.x_lhs_dirichlet_bc_func(ySpacePoints(k, 0), timePoints(timestep_no, 0));
            nextTimestepMatrix(k, nextTimestepMatrix.cols() - 1) = conditionFuncs.x_rhs_dirichlet_bc_func(ySpacePoints(k, 0), timePoints(timestep_no, 0));
        }
    }

    // Grab the previous timestep's solution matrix
    Eigen::MatrixXd previousTimestepMatrix = solutionMatrices[timestep_no - 1];

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

    double dx = 1.0/N_x;
    double dy = 1.0/N_y;
    double q = dt/pow(dx, 2.0);
    double r = dt/pow(dy, 2.0);

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
                    b(0, 0) += -2 * (dt/dx) * conditionFuncs.x_lhs_neumann_bc_func(ySpacePoints(i, 0), timePoints(timestep_no, 0));
                    b(0, 0) += q * previousTimestepMatrix(0, 0);
                }
                else if(j == xLowerBound && i > yLowerBound && i < yUpperBound)
                {
                    b(xMatricesDim * (i - yLowerBound), 0) += -2 * (dt/dx) * conditionFuncs.x_lhs_neumann_bc_func(ySpacePoints(i, 0), timePoints(timestep_no, 0));
                }
                else if(j == xLowerBound && i == yUpperBound)
                {
                    b(xMatricesDim * (yMatricesDim - 1), 0) += -2 * (dt/dx) * conditionFuncs.x_lhs_neumann_bc_func(ySpacePoints(i, 0), timePoints(timestep_no, 0));
                    b(xMatricesDim * (yMatricesDim - 1), 0) += q * previousTimestepMatrix(N_y, 0);
                }

                if(j == xUpperBound && i == yLowerBound)
                {
                    b(xMatricesDim - 1, 0) += 2 * (dt/dx) * conditionFuncs.x_rhs_neumann_bc_func(ySpacePoints(i, 0), timePoints(timestep_no, 0));
                    b(xMatricesDim - 1, 0) += q * previousTimestepMatrix(0, xMatricesDim - 1);
                }
                else if(j == xUpperBound && i > yLowerBound && i < yUpperBound)
                {
                    b(xMatricesDim * (i - yLowerBound) + (xMatricesDim - 1), 0) += 2 * (dt/dx) * conditionFuncs.x_rhs_neumann_bc_func(ySpacePoints(i, 0), timePoints(timestep_no, 0));
                }
                else if(j == xUpperBound && i == yUpperBound)
                {
                    b(xMatricesDim * yMatricesDim - 1, 0) += 2 * (dt/dx) * conditionFuncs.x_rhs_neumann_bc_func(ySpacePoints(i, 0), timePoints(timestep_no, 0));
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
                    b(0, 0) += -2 * (dt/dy) * conditionFuncs.y_lower_neumann_bc_func(xSpacePoints(j, 0), timePoints(timestep_no, 0));
                    b(0, 0) += r * previousTimestepMatrix(0, 0);
                }
                else if(i == yLowerBound && j > xLowerBound && j < xUpperBound)
                {
                    b(j - xLowerBound, 0) += -2 * (dt/dy) * conditionFuncs.y_lower_neumann_bc_func(xSpacePoints(j, 0), timePoints(timestep_no, 0));
                }
                else if(i == yLowerBound && j == xUpperBound)
                {
                    b(j - xLowerBound, 0) += -2 * (dt/dy) * conditionFuncs.y_lower_neumann_bc_func(xSpacePoints(j, 0), timePoints(timestep_no, 0));
                    b(j - xLowerBound, 0) += r * previousTimestepMatrix(0, N_x);
                }

                if(i == yUpperBound && j == xLowerBound)
                {
                    b(xMatricesDim * (yMatricesDim - 1), 0) += 2 * (dt/dy) * conditionFuncs.y_upper_neumann_bc_func(xSpacePoints(j, 0), timePoints(timestep_no, 0));
                    b(xMatricesDim * (yMatricesDim - 1), 0) += r * previousTimestepMatrix(N_y, 0);
                }
                else if(i == yUpperBound && j > xLowerBound && j < xUpperBound)
                {
                    b(xMatricesDim * (yMatricesDim - 1) + j - xLowerBound, 0) += 2 * (dt/dy) * conditionFuncs.y_upper_neumann_bc_func(xSpacePoints(j, 0), timePoints(timestep_no, 0));
                }
                else if(i == yUpperBound && j == xUpperBound)
                {
                    b(xMatricesDim * yMatricesDim - 1, 0) += 2 * (dt/dy) * conditionFuncs.y_upper_neumann_bc_func(xSpacePoints(j, 0), timePoints(timestep_no, 0));
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
                    b(xMatricesDim * (i - yLowerBound), 0) += -2 * (dt/dx) * conditionFuncs.x_lhs_neumann_bc_func(ySpacePoints(i, 0), timePoints(timestep_no, 0));
                }
                else if(j == xUpperBound - 1)
                {
                    b(xMatricesDim * (i - yLowerBound) + j, 0) += 2 * (dt/dx) * conditionFuncs.x_rhs_neumann_bc_func(ySpacePoints(i, 0), timePoints(timestep_no, 0));
                }

                if(i == yLowerBound)
                {
                    b(j - xLowerBound, 0) += -2 * (dt/dy) * conditionFuncs.y_lower_neumann_bc_func(xSpacePoints(j, 0), timePoints(timestep_no, 0));
                }
                else if(i == yUpperBound - 1)
                {
                    b(xMatricesDim * (yMatricesDim - 1) + j - xLowerBound, 0) += 2 * (dt/dy) * conditionFuncs.y_upper_neumann_bc_func(xSpacePoints(j, 0), timePoints(timestep_no, 0));
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

    if(yBCType == "neumann" && neumannBCScheme == "onesided")
    {
        for(int j = 1; j < nextTimestepMatrix.cols() - 1; j++)
        {
            nextTimestepMatrix(0, j) = nextTimestepMatrix(1, j) - dy * conditionFuncs.y_lower_neumann_bc_func(xSpacePoints(j, 0), timePoints(timestep_no, 0));
            nextTimestepMatrix(N_y, j) = nextTimestepMatrix(N_y - 1, j) + dy * conditionFuncs.y_upper_neumann_bc_func(xSpacePoints(j, 0), timePoints(timestep_no, 0));
        }
    }

    if(xBCType == "neumann" && neumannBCScheme == "onesided")
    {
        for(int k = 0; k < nextTimestepMatrix.rows(); k++)
        {
            nextTimestepMatrix(k, 0) = nextTimestepMatrix(k, 1) - dx * conditionFuncs.x_lhs_neumann_bc_func(ySpacePoints(k, 0), timePoints(timestep_no, 0));
            nextTimestepMatrix(k, N_x) = nextTimestepMatrix(k, N_x - 1) + dx * conditionFuncs.x_rhs_neumann_bc_func(ySpacePoints(k, 0), timePoints(timestep_no, 0));
        }
    }

    // Put nextTimestepMatrix into the vector of solution matrices

    solutionMatrices[timestep_no] = nextTimestepMatrix;

    // Reset nextTimestepMatrix in preparation for the next timestep

    nextTimestepMatrix.setZero();
}

void PDESolver::solve_pde()
{
    for(int n = 1; n < timePoints.size(); n++)
    {
        solve_next_timestep(n);
    }
}

void PDESolver::create_data_file(std::string filename)
{
    std::ofstream dataFile;
    dataFile.open(filename);

    for(int n = 0; n < timePoints.size(); n++)
    {
        for(int i = 0; i < solutionMatrices[n].rows(); i++)
        {
            for(int j = 0; j < solutionMatrices[n].cols(); j++)
            {
                dataFile << xSpacePoints(j, 0) << " " << ySpacePoints(i, 0) << " " << solutionMatrices[n](i, j) << std::endl;
            }
        }
    }

    dataFile.close();
}

void PDESolver::plot_solution(std::string filename)
{
    FILE* gnuplotPipe = popen("gnuplot -persist", "w");

    if(gnuplotPipe)
    {

        fprintf(gnuplotPipe, "reset\n");
        fprintf(gnuplotPipe, "set title '2D Heat Equation'\n");
        fprintf(gnuplotPipe, "set xlabel 'x'\n");
        fprintf(gnuplotPipe, "set ylabel 'y'\n");
        fprintf(gnuplotPipe, "set zlabel 'U'\n");
        fprintf(gnuplotPipe, "set dgrid3d 30,30\n");

        // Animated plot of solution over time
        fprintf(gnuplotPipe, "noOfTimePoints = %ld\n", timePoints.size());
        fprintf(gnuplotPipe, "noOfXPoints = %d\n", N_x+1);
        fprintf(gnuplotPipe, "noOfYPoints = %d\n", N_y+1);
        fprintf(gnuplotPipe, "set xrange [0:1]\nset yrange [0:1]\nset zrange [0:1.5]\n");
        fprintf(gnuplotPipe, "unset key\n");

        fprintf(gnuplotPipe, "do for [n=0:noOfTimePoints-1]{set title 'timestep '.n\nsplot \"%s\" every ::n*noOfXPoints*noOfYPoints::((n+1)*noOfXPoints*noOfYPoints-1) with lines lc rgb 'purple'\npause 0.1}\n", filename.c_str());

        fflush(gnuplotPipe);
        pclose(gnuplotPipe);

    }
}
