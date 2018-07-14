#ifndef GUARD_FiniteTimePDESolver_h
#define GUARD_FiniteTimePDESolver_h

#include <fstream>

#include "PDESolver.h"

class FiniteTimePDESolver: public PDESolver
{
    public:
        FiniteTimePDESolver(
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
                PDEConditionFunctions,
                double = 1.0/2.0
        );

        FiniteTimePDESolver(
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
                std::string,
                PDEConditionFunctions,
                double = 1.0/2.0
        );

        void create_time_points_vector();
        void get_solution_data(std::string);
        void append_next_timestep_data(std::ofstream&);
        void plot_solution(std::string);

    private:
        const double endT;
        Eigen::VectorXd timePoints;

};

#endif // GUARD_FiniteTimePDESolver_h
