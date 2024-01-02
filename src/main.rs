// #![feature(generic_const_exprs)]
// #![feature(const_trait_impl)]

pub mod stream;
pub mod consts;
mod matrix_system;
mod linalg;
mod memory;
mod direct_problem;
mod inverse_problem;
mod initial;
mod neuro;

use num::complex::{Complex64, ComplexFloat};
use num::{Complex, One, Zero};
use consts::*;
use memory::*;
use matrix_system::{fill_xy_col};
use crate::direct_problem::direct_problem;
use crate::initial::initial_k0;
use crate::inverse_problem::inverse_p;
use crate::linalg::{build_matrix, gauss};
use crate::matrix_system::{calculate_matrix_col, fill_xy_pos, fill_xyv, fxy, integral_col, r_part_vych, rpart_col};
use crate::neuro::run;
use crate::stream::{csv_complex, write_complex_vector};


fn main() {
    println!("******************************START******************************");
    println!("N = {}", N);

    println!("lambda = {} {}", (2.0*PI / K0).abs(), DIM_X / NUM_X as f64);
    println!("K0 = {:?}", K0);
    println!("Frequency = {:?}", HZ);

    let mut J = create_vector_memory(N, Complex64::new(0.1, 0.0));
    let mut K = create_vector_memory(N, K1);
    let mut W = create_vector_memory(N, Complex64::new(1.0, 0.0));
    let mut Bvych = create_vector_memory(N, Complex64::zero());
    let mut Uvych = create_vector_memory(N, Complex64::zero());
    let mut K_inv = create_vector_memory(N, Complex64::zero());
    let mut J_inv = create_vector_memory(N, Complex64::zero());



    println!("******************************DIRECT_PROBLEM******************************");
    initial_k0(N, &mut K, &mut W);

    write_complex_vector(&K, "./output/K.xls","./output/KK.xls", NUM_X, NUM_Y, N_X, N_Y);
    csv_complex("./output/K_init_r.csv", "./output/K_init_i.csv",  N, NUM_X, NUM_Y,
                N_X, N_Y, DIM_X, DIM_Y, A, B, &K);

    direct_problem(N, NUM_X, NUM_Y, N_X, N_Y, IP1, IP2, &mut W, &mut K, &mut J);

    csv_complex("./output/K_dir_r.csv", "./output/K_dir_i.csv",  N, NUM_X, NUM_Y,
                N_X, N_Y, DIM_X, DIM_Y, A, B, &K);
    csv_complex("./output/J_dir_r.csv", "./output/J_dir_i.csv",  N, NUM_X, NUM_Y,
                N_X, N_Y, DIM_X, DIM_Y, A, B, &J);
    write_complex_vector(&J, "./output/J_dir_r.xls","./output/J_dir_i.xls", NUM_X, NUM_Y, N_X, N_Y);

    let j_denoised_re = run("./output/J_dir_r.xls", MODEL,
                 "./output/J_dir_r_denoised.xls", true).unwrap().concat();
    let j_denoised_im = run("./output/J_dir_i.xls", MODEL,
                            "./output/J_dir_i_denoised.xls", true).unwrap().concat();
    //J = build_complex_vector(N, j_denoised_re, j_denoised_im);


    // Расчёт поля в точках наблюдения
    r_part_vych(point, shift, NUM_X, NUM_Y, N_X, N_Y, DIM_X, DIM_Y, A, B, K0, N, IP1, &mut Bvych);
    get_uvych(N, IP1, DIM_X, DIM_Y, &J, &mut Uvych, &Bvych, K0);

    // Обратная задача
    inverse_p(&mut Uvych, &mut J_inv, &W, &mut K_inv);


    write_complex_vector(&K_inv, "./output/K_inv.xls", "./output/KK_inv.xls", NUM_X, NUM_Y, N_X, N_Y);

    csv_complex("./output/K_inv_r.csv", "./output/K_inv_i.csv",  N, NUM_X, NUM_Y,
                        N_X, N_Y, DIM_X, DIM_Y, A, B, &K_inv);
    csv_complex("./output/J_inv_r.csv", "./output/J_inv_i.csv",  N, NUM_X, NUM_Y,
                        N_X, N_Y, DIM_X, DIM_Y, A, B, &J_inv);


}

pub fn build_complex_vector(n: usize, re: Vec<f64>, im: Vec<f64>) -> Vec<Complex<f64>> {
    assert_eq!(n, re.len());
    assert_eq!(n, im.len());
    let mut res = create_vector_memory(n, Complex64::zero());
    for (k, (r, i)) in re.into_iter().zip(im.into_iter()).enumerate() {
        res[k] = Complex::new(r, i);
    }
    res
}


pub fn get_uvych(n: usize, ip: usize, dim_x: f64, dim_y: f64, J: &Vec<Complex64>, Uvych: &mut Vec<Complex64>, Bvych: &Vec<Complex64>, k0: Complex64) {
    let n1 = NUM_X * NUM_Y;
    let A1 = vec![vec![Complex64::zero(); N]; N];

    let mut xv = vec![0f64; N];
    let mut yv = vec![0f64; N];
    let mut zv = vec![0f64; N];

    let mut x = vec![0f64; N];
    let mut y = vec![0f64; N];
    let mut z = vec![0f64; N];

    fill_xyv(n, NUM_X, NUM_Y, N_X, N_Y, dim_x, dim_y, &mut xv, &mut yv, shift);
    fill_xy_pos(point, n, NUM_X, NUM_Y, N_X, N_Y, dim_x, dim_y, A, B, &mut x, &mut y);


    for i in 0..n1 {
        Uvych[i] = Complex64::zero();
        for j in 0..n1 {
            let flag = match i == j {
                true => {1}
                false => {0}
            };
            let val = integral_col(flag, NUM_X, NUM_Y, dim_x, dim_y, A, B, ip, x[j], y[j], xv[i], yv[i], k0);

            Uvych[i] += val*J[j];
        }
        Uvych[i] += Bvych[i];
    }

    for i in n1..n {
        Uvych[i] = Complex64::zero();
        for j in n1..n {
            let flag = match i == j {
                true => {1}
                false => {0}
            };

            Uvych[i] += integral_col(flag, NUM_X, NUM_Y, dim_x, dim_y, A, B, ip, x[j], y[j], xv[i], yv[i], k0)*J[j];
        }
        Uvych[i] += Bvych[i];
    }
}



pub fn kerr(n: usize, alpha: f64, k1_: Complex64, U: &[Complex64], K: &mut [Complex64], W: &[Complex64]) {
    for p1 in 0..n {
        K[p1] = (k1_ + alpha *U[p1].norm()*U[p1]).norm()*W[p1];
    }
}

pub fn get_geometry(n: usize, W: &[Complex64]) {

}
