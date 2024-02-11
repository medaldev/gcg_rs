use num::complex::Complex64;
use num::Zero;
use crate::consts::*;
use crate::matrix_system::{fill_xy_pos, fill_xyv, fxy, integral_col};
use crate::memory::create_vector_memory;

pub fn get_uvych(point: usize, n: usize, n_x: usize, n_y: usize, ip: usize, dim_x: f64, dim_y: f64, a: f64, b: f64, shift: f64,
                 J: &Vec<Complex64>, Uvych: &mut Vec<Complex64>, Bvych: &Vec<Complex64>, k0: Complex64) {

    let mut xv = vec![0f64; n];
    let mut yv = vec![0f64; n];

    let mut x = vec![0f64; n];
    let mut y = vec![0f64; n];

    fill_xyv(n, n_x, n_y, dim_x, dim_y, a, b, &mut xv, &mut yv, shift);
    fill_xy_pos(point, n, n_x, n_y, dim_x, dim_y, a, b, &mut x, &mut y);

    for i in 0..n {
        Uvych[i] = Complex64::zero();
        for j in 0..n {
            let flag = match i == j {
                true => {1}
                false => {0}
            };

            Uvych[i] += integral_col(flag, n_x, n_y, dim_x, dim_y, a, b, ip, x[j], y[j], xv[i], yv[i], k0)*J[j];
        }
        Uvych[i] += Bvych[i];
    }
}


pub fn r_part_vych(point: usize, shift: f64, n_x: usize, n_y: usize, dim_x: f64, dim_y: f64, a: f64, b: f64, k0: Complex64, n: usize, ip: usize, Bvych: &mut Vec<Complex64>) {
    let mut xv = create_vector_memory(n, 0f64);
    let mut yv = create_vector_memory(n, 0f64);
    // let zv = vec![0f64; N];

    fill_xyv(n, n_x, n_y, dim_x, dim_y, a, b, &mut xv, &mut yv, shift);


    for i in 0..n {
        Bvych[i] = fxy(xv[i], yv[i], 0.0, k0, dim_x, dim_y);
    }
}

