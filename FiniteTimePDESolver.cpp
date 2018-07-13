#include "FiniteTimePDESolver.h"

FiniteTimePDESolver::FiniteTimePDESolver(
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
        PDEConditionFunctions conditionFuncs,
        double theta
): endT(endT),
    PDESolver(N_x, N_y, dt, startX, endX, startY, endY, startT, xBCType, yBCType, conditionFuncs, theta)
{
    create_time_points_vector();
}

FiniteTimePDESolver::FiniteTimePDESolver(
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
        PDEConditionFunctions conditionFuncs,
        double theta
): endT(endT),
    PDESolver(N_x, N_y, dt, startX, endX, startY, endY, startT, xBCType, yBCType, neumannBCScheme, conditionFuncs, theta)
{
    create_time_points_vector();
}

void FiniteTimePDESolver::create_time_points_vector()
{
    int noOfTimePoints = 0;
    double timeCounter = 0.0;

    while(timeCounter <= (endT - startT))
    {
        timeCounter += dt;
        noOfTimePoints++;
    }

    timePoints = Eigen::VectorXd::LinSpaced(noOfTimePoints, startT, endT);
}

void FiniteTimePDESolver::get_solution_data(std::string filename)
{
    std::ofstream dataFile;
    dataFile.open(filename);

    // Append the 0th timestep data to file
    append_next_timestep_data(dataFile, nextTimestepMatrix);

    // Solve all future timesteps
    for(int n = 1; n < timePoints.size(); n++)
    {
        solve_next_timestep(n);
        append_next_timestep_data(dataFile, nextTimestepMatrix);
    }

    dataFile.close();
}

void FiniteTimePDESolver::append_next_timestep_data(std::ofstream& dataFile, Eigen::MatrixXd A)
{
    for(int i = 0; i < A.rows(); i++)
    {
        for(int j = 0; j < A.cols(); j++)
        {
            dataFile << xSpacePoints(j, 0) << " " << ySpacePoints(i, 0) << " " << A(i, j) << std::endl;
        }
    }
}

void FiniteTimePDESolver::plot_solution(std::string filename)
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
