use num::complex::Complex64;
use num::Zero;
use crate::consts::{A, B, DIM_X, DIM_Y, K0, point};
use crate::linalg::{build_matrix, gauss};
use crate::matrix_system::{calculate_matrix_col, rpart_col};
use crate::memory::{create_matrix_memory, create_vector_memory};

pub fn direct_problem
(
    n: usize, num_x: usize, num_y: usize, n_x: usize, n_y: usize, ip1: usize, ip2: usize,
    W: &mut Vec<Complex64>, K: &mut Vec<Complex64>, J: &mut Vec<Complex64>,
)
{
    let mut AA = create_matrix_memory(n, Complex64::zero());
    let mut BB = create_vector_memory(n, Complex64::zero());


    rpart_col(n, num_x, num_y, n_x, n_y, DIM_X, DIM_Y, A, B, K0, ip1, ip2, &mut BB);

    calculate_matrix_col(point, num_x, num_y, n_x, n_y, K, DIM_X, DIM_Y, A, B, n, ip1, ip2, &mut AA, K0);
    build_matrix(n, &mut AA, W, &mut BB);

    for i in 0..n {
        J[i] = Complex64::zero();
    }

    gauss(n, &AA, &BB, &W, J);

}

