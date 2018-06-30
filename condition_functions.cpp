#include "condition_functions.h"

double ic_func(double x, double y)
{
    return sin(x) + cos(y);
}

double x_lhs_dirichlet_bc_func(double y, double t)
{
    return 0.0;
}

double x_rhs_dirichlet_bc_func(double y, double t)
{
    return 0.0;
}

double y_lower_dirichlet_bc_func(double x, double t)
{
    return 0.0;
}

double y_upper_dirichlet_bc_func(double x, double t)
{
    return 0.0;
}

double x_lhs_neumann_bc_func(double y, double t)
{
    return sin((y * M_PI)/t);
}

double x_rhs_neumann_bc_func(double y, double t)
{
    return sin((y * M_PI)/t);
}

double y_lower_neumann_bc_func(double x, double t)
{
    return sin((x * M_PI)/t);
}

double y_upper_neumann_bc_func(double x, double t)
{
    return sin((x * M_PI)/t);
}
