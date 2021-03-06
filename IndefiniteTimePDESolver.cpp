#include "IndefiniteTimePDESolver.h"

IndefiniteTimePDESolver::IndefiniteTimePDESolver(
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
): timestepNumber(1),
    PDESolver(N_x, N_y, dt, startX, endX, startY, endY, startT, xBCType, yBCType, conditionFuncs, theta)
{}

IndefiniteTimePDESolver::IndefiniteTimePDESolver(
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
): timestepNumber(1),
    PDESolver(N_x, N_y, dt, startX, endX, startY, endY, startT, xBCType, yBCType, neumannBCScheme, conditionFuncs, theta)
{}

Eigen::MatrixXd IndefiniteTimePDESolver::get_next_timestep_matrix()
{
    solve_next_timestep(timestepNumber);
    increment_timestep_counter();
    return nextTimestepMatrix;
}
