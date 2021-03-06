#ifndef GUARD_PDESolver_h
#define GUARD_PDESolver_h

#include <eigen3/Eigen/Eigen>
#include <vector>
#include <string>

struct PDEConditionFunctions
{
    double ((*ic_func)(double, double));
    double ((*x_lhs_dirichlet_bc_func)(double, double));
    double ((*x_rhs_dirichlet_bc_func)(double, double));
    double ((*y_lower_dirichlet_bc_func)(double, double));
    double ((*y_upper_dirichlet_bc_func)(double, double));
    double ((*x_lhs_neumann_bc_func)(double, double));
    double ((*x_rhs_neumann_bc_func)(double, double));
    double ((*y_lower_neumann_bc_func)(double, double));
    double ((*y_upper_neumann_bc_func)(double, double));
};

class PDESolver
{
    public:
        PDESolver(
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

        PDESolver(
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

        void initialise_vectors_matrices();
        void create_discretisation_vectors();
        void create_solution_matrix();
        void perform_prelim_mathematical_routines();
        void apply_prelim_ic_bcs();
        void get_solution_looping_bounds();
        void create_kronecker_product_matrices();
        Eigen::MatrixXd kronecker_product(Eigen::MatrixXd, Eigen::MatrixXd);
        void solve_next_timestep(int);

    protected:
        const int N_x;
        const int N_y;
        const double dt;

        const double startX;
        const double endX;
        const double startY;
        const double endY;
        const double startT;

        const std::string xBCType;
        const std::string yBCType;
        const std::string neumannBCScheme;
        PDEConditionFunctions conditionFuncs;

        const double theta;

        Eigen::VectorXd xSpacePoints;
        Eigen::VectorXd ySpacePoints;

        Eigen::MatrixXd nextTimestepMatrix;

        int xMatricesDim;
        int yMatricesDim;

        Eigen::MatrixXd previousTimestepKronProdMatrixA;
        Eigen::MatrixXd previousTimestepKronProdMatrixB;
        Eigen::MatrixXd nextTimestepKronProdMatrixA;
        Eigen::MatrixXd nextTimestepKronProdMatrixB;

        int xLowerBound;
        int xUpperBound;
        int yLowerBound;
        int yUpperBound;
};

#endif //GUARD_PDESolver_h
