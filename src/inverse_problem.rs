use num::complex::Complex64;
use num::Zero;
use crate::consts::*;
use crate::linalg::gauss;
use crate::matrix_system::{fill_xy_pos, fill_xyv, fxy, integral_col};
use crate::memory::{create_matrix_memory, create_vector_memory};

pub fn inverse_p(n: usize, n_x: usize, n_y: usize, point: usize, shift: f64, ip: usize, a: f64, b: f64, dim_x: f64, dim_y: f64, k0: Complex64,
                 Uvych: &Vec<Complex64>, J1: &mut Vec<Complex64>, KKK: &mut Vec<Complex64>) {
    println!("\n******************************INVERSE_PROBLEM*****************************");

    get_jj(n, n_x, n_y, point, shift, ip, a, b, dim_x, dim_y, k0, J1, Uvych);
    get_k1(n, n_x, n_y, point, ip, a, b, dim_x, dim_y, k0, KKK, J1);

}

pub fn get_jj(n: usize, n_x: usize, n_y: usize, point: usize, shift: f64, ip: usize, a: f64, b: f64, dim_x: f64, dim_y: f64, k0: Complex64, J: &mut Vec<Complex64>, Uvych: &Vec<Complex64>) {
    let mut A1 = create_matrix_memory(n, Complex64::zero());
    let mut B1 = create_vector_memory(n, Complex64::zero());
    let W1 = create_vector_memory(n, Complex64::new(1.0, 0.0));


    let mut xv = vec![0f64; n];
    let mut yv = vec![0f64; n];

    let mut x = vec![0f64; n];
    let mut y = vec![0f64; n];

    fill_xyv(n, n_x, n_y, dim_x, dim_y, &mut xv, &mut yv, shift);
    fill_xy_pos(point, n, n_x, n_y, dim_x, dim_y, a, b, &mut x, &mut y);

    for i in 0..n {
        for j in 0..n {
            let flag = match i == j {
                true => {1}
                false => {0}
            };

            A1[i][j] = integral_col(flag, n_x, n_y, dim_x, dim_y, a, b, ip, x[j], y[j], xv[i], yv[i], k0);
        }
    }

    for i in 0..n {
        B1[i] = Uvych[i] - fxy(xv[i], yv[i], 0.0, k0, dim_x, dim_y);
    }

    gauss(n, &A1, &B1, &W1, J);



}

pub fn get_k1(n: usize, n_x: usize, n_y: usize, point: usize, ip: usize, a: f64, b: f64, dim_x: f64, dim_y: f64, k0: Complex64, K: &mut Vec<Complex64>, J: &mut Vec<Complex64>) {

    let mut A1 = create_vector_memory(n, Complex64::zero());

    let l_x = dim_x / (n_x as f64);
    let l_y = dim_y / (n_y as f64);

    let mut x = vec![0f64; n];
    let mut y = vec![0f64; n];
    let mut z = vec![0f64; n];

    fill_xy_pos(point, n, n_x, n_y, dim_x, dim_y, a, b, &mut x, &mut y);

    for i in 0..n {
        A1[i] = Complex64::zero();
        for j in 0..n {
            let flag = match i == j {
                true => {1}
                false => {0}
            };

            A1[i] += integral_col(flag, n_x, n_y, dim_x, dim_y, a, b, ip, x[j], y[j], x[i] + l_x / 2.0, y[i] + l_y / 2.0, k0)*J[j];
        }
    }


    for i in 0..n {
        K[i] = J[i]  / (A1[i] + fxy(x[i] + l_x / 2.0, y[i] + l_y / 2.0, 0.0, k0, dim_x, dim_y));
    }


}

