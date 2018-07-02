#ifndef GUARD_PDESolver_h
#define GUARD_PDESolver_h

#include <eigen3/Eigen/Eigen>
#include <vector>
#include <string>
#include <fstream>

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
                double,
                std::string,
                std::string,
                PDEConditionFunctions
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
                double,
                std::string,
                std::string,
                std::string,
                PDEConditionFunctions
        );

        void initialise_vectors_matrices();
        void create_discretisation_vectors();
        void create_solution_matrices();
        void perform_prelim_mathematical_routines();
        void apply_prelim_ic_bcs();
        void get_solution_looping_bounds();
        void create_kronecker_product_matrices();
        Eigen::MatrixXd kronecker_product(Eigen::MatrixXd, Eigen::MatrixXd);
        void solve_next_timestep(int);
        void get_solution_data(std::string);
        void append_next_timestep_data(std::ofstream&, Eigen::MatrixXd);
        void plot_solution(std::string);

    private:
        const int N_x;
        const int N_y;
        const double dt;

        const double startX;
        const double endX;
        const double startY;
        const double endY;
        const double startT;
        const double endT;

        const std::string xBCType;
        const std::string yBCType;
        const std::string neumannBCScheme;
        PDEConditionFunctions conditionFuncs;

        Eigen::VectorXd xSpacePoints;
        Eigen::VectorXd ySpacePoints;
        Eigen::VectorXd timePoints;

        Eigen::MatrixXd U;
        Eigen::MatrixXd nextTimestepMatrix;

        int xMatricesDim;
        int yMatricesDim;

        Eigen::MatrixXd kroneckerProdMatrixA;
        Eigen::MatrixXd kroneckerProdMatrixB;

        int xLowerBound;
        int xUpperBound;
        int yLowerBound;
        int yUpperBound;
};

#endif //GUARD_PDESolver_h
