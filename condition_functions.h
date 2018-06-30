#ifndef GUARD_condition_functions_h
#define GUARD_condition_functions_h

#include <cmath>

double ic_func(double, double);

double x_lhs_dirichlet_bc_func(double, double);

double x_rhs_dirichlet_bc_func(double, double);

double y_lower_dirichlet_bc_func(double, double);

double y_upper_dirichlet_bc_func(double, double);

double x_lhs_neumann_bc_func(double, double);

double x_rhs_neumann_bc_func(double, double);

double y_lower_neumann_bc_func(double, double);

double y_upper_neumann_bc_func(double, double);

#endif // GUARD_condition_functions_h
