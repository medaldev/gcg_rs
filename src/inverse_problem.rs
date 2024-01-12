use num::complex::Complex64;
use num::Zero;
use crate::consts::*;
use crate::linalg::gauss;
use crate::matrix_system::{fill_xy_pos, fill_xyv, fxy, integral_col};
use crate::memory::{create_matrix_memory, create_vector_memory};

pub fn inverse_p(Uvych: &Vec<Complex64>, J1: &mut Vec<Complex64>, W: &Vec<Complex64>, KKK: &mut Vec<Complex64>) {
    println!("\n******************************INVERSE_PROBLEM*****************************");
    get_jj(N, NUM_X, NUM_Y, N_X, N_Y, IP1, A, B, DIM_X, DIM_Y, K0, W, J1, Uvych);

    //WriteVector(J1, "J_inv.xls", "JJ_inv.xls", NUM_X, NUM_Y, n_x, n_y);

    get_k1(N, IP1, A, B, DIM_X, DIM_Y, K0, KKK, J1);

}

pub fn get_jj(n: usize, num_x: usize, num_y: usize, n_x: usize, n_y: usize, ip: usize, a: f64, b: f64, dim_x: f64, dim_y: f64, k0: Complex64, W: &Vec<Complex64>, J: &mut Vec<Complex64>, Uvych: &Vec<Complex64>) {
    let n1 = num_x * num_y;
    let mut A1 = create_matrix_memory(n, Complex64::zero());
    let mut B1 = create_vector_memory(n, Complex64::zero());
    let mut W1 = create_vector_memory(n, Complex64::new(1.0, 0.0));

    let mut xv = vec![0f64; N];
    let mut yv = vec![0f64; N];

    let mut x = vec![0f64; N];
    let mut y = vec![0f64; N];

    fill_xyv(n, num_x, num_y, n_x, n_y, dim_x, dim_y, &mut xv, &mut yv, SHIFT);
    fill_xy_pos(POINT, n, num_x, num_y, n_x, n_y, dim_x, dim_y, a, b, &mut x, &mut y);

    for i in 0..n1 {
        for j in 0..n1 {
            let flag = match i == j {
                true => {1}
                false => {0}
            };

            A1[i][j] = integral_col(flag, num_x, num_y, dim_x, dim_y, a, b, ip, x[j], y[j], xv[i], yv[i], k0);
        }
    }

    for i in n1..n {
        for j in n1..n {
            let flag = match i == j {
                true => {1}
                false => {0}
            };

            A1[i][j] = integral_col(flag, num_x, num_y, dim_x, dim_y, a, b, ip, x[j], y[j], xv[i], yv[i], k0);
        }
    }

    for i in 0..n {
        B1[i] = Uvych[i] - fxy(xv[i], yv[i], 0.0, k0, dim_x, dim_y);
    }

    gauss(n, &A1, &B1, &W1, J);



}

pub fn get_k1(n: usize, ip: usize, a: f64, b: f64, dim_x: f64, dim_y: f64, k0: Complex64, K: &mut Vec<Complex64>, J: &mut Vec<Complex64>) {
    let n1 = NUM_X * NUM_Y;

    let len_x = dim_x / NUM_X as f64;
    let len_y = dim_y / NUM_Y as f64;

    let mut A1 = create_vector_memory(n, Complex64::zero());

    let mut x = vec![0f64; N];
    let mut y = vec![0f64; N];
    let mut z = vec![0f64; N];

    fill_xy_pos(POINT, n, NUM_Y, NUM_Y, N_X, N_Y, dim_x, dim_y, a, b, &mut x, &mut y);

    for i in 0..n1 {
        A1[i] = Complex64::zero();
        for j in 0..n1 {
            let flag = match i == j {
                true => {1}
                false => {0}
            };

            A1[i] += integral_col(flag, NUM_X, NUM_Y, dim_x, dim_y, a, b, ip, x[j], y[j], x[i] + len_x / 2.0, y[i] + len_y / 2.0, k0)*J[j];
        }
    }

    for i in n1..n {
        A1[i] = Complex64::zero();
        for j in n1..n {
            let flag = match i == j {
                true => {1}
                false => {0}
            };

            A1[i] += integral_col(flag, NUM_X, NUM_Y, dim_x, dim_y, a, b, ip, x[j], y[j], x[i] + len_x / 2.0, y[i] + len_y / 2.0, k0)*J[j];
        }
    }

    for i in 0..N {
        K[i] = J[i]  / (A1[i] + fxy(x[i] + len_x / 2.0, y[i] + len_y / 2.0, 0.0, k0, dim_x, dim_y));
    }


}

