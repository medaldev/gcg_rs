use num::complex::Complex64;
use num::Zero;
use crate::linalg::{build_matrix, gauss};
use crate::matrix_system::{calculate_matrix_col, rpart_col};
use crate::memory::{create_matrix_memory, create_vector_memory};

pub fn direct_problem
(
    point: usize, n: usize, n_x: usize, n_y: usize, dim_x: f64, dim_y: f64, a: f64, b: f64, k0: Complex64, ip1: usize, ip2: usize,
    W: &mut Vec<Complex64>, K: &mut Vec<Complex64>, J: &mut Vec<Complex64>,
)
{
    let mut AA = create_matrix_memory(n, Complex64::zero());
    let mut BB = create_vector_memory(n, Complex64::zero());


    rpart_col(n, n_x, n_y, dim_x, dim_y, a, b, k0, ip1, ip2, &mut BB);

    calculate_matrix_col(point, n_x, n_y, K, dim_x, dim_y, a, b, n, ip1, ip2, &mut AA, k0);
    build_matrix(n, &mut AA, W, &mut BB);

    for i in 0..n {
        J[i] = Complex64::zero();
    }

    gauss(n, &AA, &BB, &W, J);

}

