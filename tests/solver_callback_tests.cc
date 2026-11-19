#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/vector.h>
#include <iostream>

// A success-returning observer must neither stop CG early nor suppress failure.
int main()
{
    using namespace dealii;
    FullMatrix<double> matrix(3, 3);
    for (unsigned int i = 0; i < 3; ++i)
        matrix(i, i) = i + 1;
    Vector<double> rhs(3), solution(3), residual(3);
    rhs = 1;
    unsigned int calls = 0;
    SolverControl control(10, 1e-12);
    SolverCG<Vector<double>> solver(control);
    solver.connect([&calls](unsigned int, double, const Vector<double> &) {
        ++calls;
        return SolverControl::success;
    });
    solver.solve(matrix, solution, rhs, PreconditionIdentity());
    matrix.vmult(residual, solution);
    residual -= rhs;
    AssertThrow(control.last_step() == 3 && calls == 4 && residual.l2_norm() < 1e-12,
                ExcMessage("Observer changed convergence behavior"));

    solution = 0;
    SolverControl limited(1, 1e-12);
    SolverCG<Vector<double>> failing_solver(limited);
    failing_solver.connect([](unsigned int, double, const Vector<double> &) {
        return SolverControl::success;
    });
    bool failed = false;
    try {
        failing_solver.solve(matrix, solution, rhs, PreconditionIdentity());
    } catch (const SolverControl::NoConvergence &) {
        failed = true;
    }
    AssertThrow(failed, ExcMessage("Observer suppressed iteration-limit failure"));
    std::cout << "Solver callback convergence and failure checks passed\n";
}
