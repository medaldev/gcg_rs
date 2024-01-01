// #![feature(generic_const_exprs)]
// #![feature(const_trait_impl)]

pub mod stream;
pub mod consts;
mod matrix_system;
mod linalg;
mod memory;

use num::complex::{Complex64, ComplexFloat};
use num::{Complex, One, Zero};
use consts::*;
use memory::*;
use matrix_system::{fill_xy_col};
use crate::linalg::{build_matrix, gauss};
use crate::matrix_system::{calculate_matrix_col, fill_xy_pos, fill_xyv, fxy, integral_col, r_part_vych, rpart_col};
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


pub fn inverse_p(Uvych: &mut Vec<Complex64>, J1: &mut Vec<Complex64>, W: &Vec<Complex64>, KKK: &mut Vec<Complex64>) {
    println!("********************RUN_INVERSE_PROBLEM********************");
    get_jj(N, NUM_X, NUM_Y, N_X, N_Y, IP1, A, B, DIM_X, DIM_Y, K0, W, J1, Uvych);

    //WriteVector(J1, "J_inv.xls", "JJ_inv.xls", NUM_X, NUM_Y, n_x, n_y);

    get_k1(N, IP1, A, B, DIM_X, DIM_Y, K0, KKK, J1);

}

pub fn get_jj(n: usize, num_x: usize, num_y: usize, n_x: usize, n_y: usize, ip: usize, a: f64, b: f64, dim_x: f64, dim_y: f64, k0: Complex64, W: &Vec<Complex64>, J: &mut Vec<Complex64>, Uvych: &mut Vec<Complex64>) {
    let n1 = num_x * num_y;
    let mut A1 = create_matrix_memory(n, Complex64::zero());
    let mut B1 = create_vector_memory(n, Complex64::zero());
    let mut W1 = create_vector_memory(n, Complex64::new(1.0, 0.0));

    let mut xv = vec![0f64; N];
    let mut yv = vec![0f64; N];

    let mut x = vec![0f64; N];
    let mut y = vec![0f64; N];

    fill_xyv(n, num_x, num_y, n_x, n_y, dim_x, dim_y, &mut xv, &mut yv, shift);
    fill_xy_pos(point, n, num_x, num_y, n_x, n_y, dim_x, dim_y, a, b, &mut x, &mut y);

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

    fill_xy_pos(point, n, NUM_Y, NUM_Y, N_X, N_Y, dim_x, dim_y, a, b, &mut x, &mut y);

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

    // println!("W{:?}", W);
    //
    // println!("J{:?}", J);
    //
    // std::process::exit(0);

}



pub fn kerr(n: usize, alpha: f64, k1_: Complex64, U: &[Complex64], K: &mut [Complex64], W: &[Complex64]) {
    for p1 in 0..n {
        K[p1] = (k1_ + alpha *U[p1].norm()*U[p1]).norm()*W[p1];
    }
}

pub fn get_geometry(n: usize, W: &[Complex64]) {

}


pub fn initial_k0(n: usize, K: &mut [Complex64], W: &mut [Complex64]) {

    let (mut p1, mut p2, mut p3, mut p4) = (0, 0, 0, 0);
    let (mut x, mut y) = (0.0, 0.0);
    let mut s = Complex64::zero();

    let mut xc = create_vector_memory(n, 0.0f64);
    let mut yc = create_vector_memory(n, 0.0f64);

    fill_xy_col(n, NUM_X, NUM_Y, N_X, N_Y, DIM_X, DIM_Y, A, B, &mut xc, &mut yc);

    let r1 = DIM_X / 2.0;
    let r2 = 3.* DIM_X / 8.0;
    let r3 = DIM_X / 4.0;
    let r4 = DIM_X / 8.0;
    let mut ind = 0;

    for i in 0..N_X {
        for j in 0..N_Y {

            x = xc[N1 + ind];
            y = yc[N1 + ind];

            K[N1 + ind] = Complex64::new(0.7, 0.);
            if x < 0.0 {
                K[N1 + ind] = Complex64::new(0.6, 0.);
            }

            W[N1 + ind] = Complex64::new(1.0, 0.);

            ind += 1;
        }
    }

    // ind = 0;
    // for i in 0..N_X {
    //     for j in 0..N_Y {
    //
    //         x = xc[N1 + ind];
    //         y = yc[N1 + ind];
    //
    //         if x * x + y * y <= r1 * r1 {
    //             K[N1 + ind] = Complex64::new(0.7, 0.);
    //             if (x < 0.){
    //                 K[N1 + ind] = Complex64::new(0.6, 0.);
    //             }
    //
    //             W[N1 + ind] = Complex64::new(1.0, 0.);
    //
    //             if (x * x + y * y <= r2 * r2) {
    //                 K[N1 + ind] = K1;
    //                 W[N1 + ind] = Complex64::zero();
    //
    //                 if (x * x + y * y <= r3 * r3) {
    //                     K[N1 + ind] = Complex64::new(0.35, 0.);
    //                     if (x < 0.){
    //                         K[N1 + ind] = Complex64::new(0.25, 0.);
    //                     }
    //
    //                     W[N1 + ind] = Complex64::new(1.0, 0.);
    //                     if (x * x + y * y <= r4 * r4) {
    //                         K[N1 + ind] = K1;
    //                         W[N1 + ind] = Complex64::zero();
    //                     }
    //
    //                 }
    //             }
    //         }
    //         else {
    //             K[N1 + ind] = K1;
    //             W[N1 + ind] = Complex64::zero();
    //         }
    //
    //         ind += 1;
    //     }
    // }

    ind = 0;
    for i2 in 0..NUM_Y {
        for i3 in 0..NUM_X {

            p1 = N1 + 2 * i2* N_X + 2 * i3;

            p2 = p1 + 1;
            p3 = p1 + N_X;
            p4 = p1 + N_X + 1;

            s = K[p1] + K[p2] + K[p3] + K[p4];
            K[ind] = s / (point * point) as f64;

            ind += 1;

        }
    }

    for i1 in 0..N {
        K[i1] *= W[i1];
    }

}





/*


void kerr(int n, double alpha, mycomplex K1, mycomplex *U, mycomplex *K, mycomplex *W){
	int p1/*, p2, p3*/;

	for (p1 = 0; p1 < n; p1++){
		K[p1] = (K1 + alpha*abs(U[p1])*abs(U[p1]))*W[p1];
	}
}
 */


