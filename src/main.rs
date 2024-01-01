#![feature(generic_const_exprs)]

pub mod stream;
pub mod consts;
mod matrix_system;
mod linalg;
mod memory;

use num::complex::{Complex64, ComplexFloat};
use num::{Complex, One, Zero};
use consts::*;
use matrix_system::{fill_xy_col};
use crate::linalg::{build_matrix, gauss};
use crate::matrix_system::{calculate_matrix_col, fill_xy_pos, fill_xyv, fxy, integral_col, r_part_vych, rpart_col};


fn main() {
    println!("******************************START******************************");
    println!("N = {}", N);

    println!("lambda = {} {}", (2.0*PI / K0).abs(), DIM_X / NUM_X as f64);
    println!("K0 = {:?}", K0);
    println!("Frequency = {:?}", HZ);

    let mut J = vec![Complex64::new(0.1, 0.0); N];
    let mut K = vec![K1; N];
    let mut W = vec![Complex64::new(1.0, 0.0); N];
    let mut Bvych = vec![Complex64::zero(); N];
    let mut Uvych = vec![Complex64::zero(); N];
    let mut K_inv = vec![Complex64::zero(); N];
    let mut J_inv = vec![Complex64::zero(); N];

    println!("******************************DIRECT_PROBLEM******************************");
    initial_k0::<N>(K.as_mut(), W.as_mut());

    // WriteVector(K, "K.xls", "KK.xls", NUM_X, NUM_Y, n_x, n_y);
    // CSVmycomplex("K_init_r.csv", "K_init_i.csv", N, NUM_X, NUM_Y, n_x, n_y, dim_x, DIM_Y, a, b, K);
    //

    direct_problem::<N, NUM_X, NUM_Y, N_X, N_Y, ip1, ip2>(&mut W, &mut K, &mut J);
    r_part_vych(point, shift, NUM_X, NUM_Y, N_X, N_Y, DIM_X, DIM_Y, A, B, K0, N, ip1, &mut Bvych);
    get_uvych(N, ip1, DIM_X, DIM_Y, &J, &mut Uvych, &mut Bvych, K0);
    inverse_p(&mut Uvych, &mut J_inv, &W, &mut K_inv);
    //DirectProblem(NUM_X, NUM_Y, n_x, n_y, ip1, ip2, N, W, K, J);

    // CSVmycomplex("K_dir_r.csv", "K_dir_i.csv", N, NUM_X, NUM_Y, n_x, n_y, dim_x, DIM_Y, a, b, K);
    // CSVmycomplex("J_dir_r.csv", "J_dir_i.csv", N, NUM_X, NUM_Y, n_x, n_y, dim_x, DIM_Y, a, b, J);
    //
    // RpartVych(point, shift, NUM_X, NUM_Y, n_x, n_y, dim_x, DIM_Y, a, b, K0, N, ip1, Bvych);
    // Get_Uvych(N, ip1, dim_x, DIM_Y, J, Uvych, Bvych, K0);
    //
    // // �������� ������
    // InverseP(Uvych, J_inv, W, K_inv);
    // WriteVector(K_inv, "K_inv.xls", "KK_inv.xls", NUM_X, NUM_Y, n_x, n_y);
    // CSVmycomplex("K_inv_r.csv", "K_inv_i.csv", N, NUM_X, NUM_Y, n_x, n_y, dim_x, DIM_Y, a, b, K_inv);
    // CSVmycomplex("J_inv_r.csv", "J_inv_i.csv", N, NUM_X, NUM_Y, n_x, n_y, dim_x, DIM_Y, a, b, J_inv);

}


pub fn inverse_p(Uvych: &mut Vec<Complex64>, J1: &mut Vec<Complex64>, W: &Vec<Complex64>, KKK: &mut Vec<Complex64>) {
    println!("********************RUN_INVERSE_PROBLEM********************");
    get_jj(N, NUM_X, NUM_Y, N_X, N_Y, ip1, A, B, DIM_X, DIM_Y, K0, W, J1, Uvych);
    //WriteVector(J1, "J_inv.xls", "JJ_inv.xls", NUM_X, NUM_Y, n_x, n_y);

    get_k1(N, ip1, A, B, DIM_X, DIM_Y, K0, KKK, J1);
}

pub fn get_jj(n: usize, num_x: usize, num_y: usize, n_x: usize, n_y: usize, ip: usize, a: f64, b: f64, dim_x: f64, dim_y: f64, k0: Complex64, W: &Vec<Complex64>, J: &mut Vec<Complex64>, Uvych: &mut Vec<Complex64>) {
    let n1 = num_x * num_y;
    let mut A1 = vec![vec![Complex64::zero(); N]];
    let A11 = vec![vec![Complex64::zero(); N]];
    let A2 = vec![vec![Complex64::zero(); N]];
    let A22 = vec![vec![Complex64::zero(); N]];
    let A3 = vec![vec![Complex64::zero(); N]];
    let A33 = vec![vec![Complex64::zero(); N]];
    let EE = vec![vec![Complex64::zero(); N]];
    let mut B1 = vec![Complex64::zero(); N];
    let B2 = vec![Complex64::zero(); N];
    let B3 = vec![Complex64::zero(); N];
    let mut W1 = vec![Complex64::new(1.0, 0.0); N];
    let Uvych_New = vec![Complex64::zero(); N];

    let mut xv = vec![0f64; N];
    let mut yv = vec![0f64; N];
    let mut zv = vec![0f64; N];

    let mut x = vec![0f64; N];
    let mut y = vec![0f64; N];
    let mut z = vec![0f64; N];

    fill_xyv(n, num_x, num_y, n_x, n_y, dim_x, dim_y, &mut xv, &mut yv, shift);
    fill_xy_pos(point, n, num_x, num_y, n_x, n_y, dim_x, dim_y, a, b, &mut x, &mut y);

    for i in 0..n1 {
        for j in 0..n1 {
            let flag = match i == j {
                true => {1}
                false => {0}
            };

            A1[i][j] = integral_col(flag, NUM_X, NUM_Y, dim_x, dim_y, a, b, ip, x[j], y[j], xv[i], yv[i], k0);
        }
    }

    for i in n1..n {
        Uvych[i] = Complex64::zero();
        for j in n1..n {
            let flag = match i == j {
                true => {1}
                false => {0}
            };

            A1[i][j] = integral_col(flag, NUM_X, NUM_Y, dim_x, dim_y, a, b, ip, x[j], y[j], xv[i], yv[i], k0);
        }
    }

    for i in 0..N {
        B1[i] = Uvych[i] - fxy(xv[i], yv[i], 0.0, k0, dim_x, dim_y);
    }

    gauss(N, &A1, &B1, &W1, J);



}

pub fn get_k1(n: usize, ip: usize, a: f64, b: f64, dim_x: f64, dim_y: f64, k0: Complex64, K: &mut Vec<Complex64>, J: &mut Vec<Complex64>) {
    let n1 = NUM_X * NUM_Y;

    let len_x = dim_x / NUM_X as f64;
    let len_y = dim_y / NUM_Y as f64;

    let mut A1 = vec![Complex64::zero(); N];

    let mut x = vec![0f64; N];
    let mut y = vec![0f64; N];
    let mut z = vec![0f64; N];

    fill_xy_pos(point, N, NUM_Y, NUM_Y, N_X, N_Y, dim_x, dim_y, a, b, &mut x, &mut y);

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



pub fn get_uvych(n: usize, ip: usize, dim_x: f64, dim_y: f64, J: &Vec<Complex64>, Uvych: &mut Vec<Complex64>, Bvych: &mut Vec<Complex64>, k0: Complex64) {
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

            Uvych[i] += integral_col(flag, NUM_X, NUM_Y, dim_x, dim_y, A, B, ip, x[j], y[j], xv[i], yv[i], k0)*J[j];
        }
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
    }
}

pub fn direct_problem<
    const N: usize,
    const num_x: usize,
    const num_y: usize,
    const n_x: usize,
    const n_y: usize,
    const ip1: usize,
    const ip2: usize,
>
(
    W: &mut Vec<Complex64>, K: &mut Vec<Complex64>, J: &mut Vec<Complex64>,
)
{
    let mut AA = vec![vec![Complex64::zero(); N]; N];
    let mut BB = vec![Complex64::zero(); N];
    rpart_col(N, num_x, num_y, n_x, n_y, DIM_X, DIM_Y, A, B, K0, ip1, ip2, &mut BB);

    calculate_matrix_col(N, num_x, num_y, point, n_x, n_y, ip1, ip2, DIM_X, DIM_Y, A, B, K, &mut AA, K0);

    build_matrix(N, &mut AA, W, &mut BB);

    gauss(N, &AA, &mut BB, &W, J);




}



pub fn kerr(n: usize, alpha: f64, k1_: Complex64, U: &[Complex64], K: &mut [Complex64], W: &[Complex64]) {
    for p1 in 0..n {
        K[p1] = (k1_ + Complex64::from(alpha) *U[p1].norm()*U[p1]).norm()*W[p1];
    }
}

pub fn get_geometry(n: usize, W: &[Complex64]) {

}


pub fn initial_k0<const N: usize>(K: &mut [Complex64], W: &mut [Complex64]) {

    let (mut p1, mut p2, mut p3, mut p4) = (0, 0, 0, 0);
    let (mut x, mut y) = (0.0, 0.0);
    let mut s = Complex64::zero();

    let mut xc = [0.0f64; N];
    let mut yc = [0.0f64; N];

    fill_xy_col(N, NUM_X, NUM_Y, N_X, N_Y, DIM_X, DIM_Y, A, B, xc.as_mut(), yc.as_mut());

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


