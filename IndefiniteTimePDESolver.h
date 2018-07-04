#ifndef GUARD_IndefiniteTimePDESolver_h
#define GUARD_IndefiniteTimePDESolver_h

#include "PDESolver.h"

class IndefiniteTimePDESolver: public PDESolver
{
    public:
        IndefiniteTimePDESolver(
                int,
                int,
                double,
                double,
                double,
                double,
                double,
                double,
                std::string,
                std::string,
                PDEConditionFunctions
        );

        IndefiniteTimePDESolver(
                int,
                int,
                double,
                double,
                double,
                double,
                double,
                double,
                std::string,
                std::string,
                std::string,
                PDEConditionFunctions
        );

        void increment_timestep_counter(){timestepNumber++;};
        void reset_timestep_counter(){timestepNumber = 1;};
        Eigen::MatrixXd get_next_timestep_matrix();

    private:
        int timestepNumber;
};

#endif // GUARD_IndefiniteTimePDESolver_h
