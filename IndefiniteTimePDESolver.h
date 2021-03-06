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
                PDEConditionFunctions,
                double = 1.0/2.0
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
                PDEConditionFunctions,
                double = 1.0/2.0
        );

        void increment_timestep_counter(){timestepNumber++;};
        void reset_timestep_counter(){timestepNumber = 1;};
        Eigen::MatrixXd get_next_timestep_matrix();
        void set_next_timestep_matrix_entry(int i, int j, double (*fp)(double)){nextTimestepMatrix(j, i) = (*fp)(nextTimestepMatrix(j, i));};

    private:
        int timestepNumber;
};

#endif // GUARD_IndefiniteTimePDESolver_h
