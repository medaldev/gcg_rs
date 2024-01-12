use num::complex::{Complex64, ComplexFloat};
use num::Zero;
use crate::memory::create_vector_memory;

pub fn fill_xyv(n: usize, n_x: usize, n_y: usize, dim_x: f64, dim_y: f64, xv: &mut [f64], yv: &mut [f64], shift: f64) {

    let mut p = 0;

    let l_x = dim_x / (n_x as f64);
    let l_y = dim_y / (n_y as f64);

    for i in 0..n_y {
        for j in 0..n_x {
            if j < n_x / 2 {
                xv[p] = -dim_x / 2.0 - (j as f64)*l_x - l_x / 2.0 - l_x / 20.0;
            } else {
                xv[p] = dim_x / 2.0 + (j as f64)*l_x + l_x / 2.0 + l_x / 20.0;
            }
            yv[p] = -dim_y / 2.0 + (i as f64)*l_y + l_y / 2.0;

            p += 1;
        }
    }
}


pub fn fill_xy_pos(point: usize, n: usize, n_x: usize, n_y: usize, dim_x: f64, dim_y: f64, a: f64, b: f64, x: &mut [f64], y: &mut [f64]) {
    let l_x = dim_x / (n_x as f64);
    let l_y = dim_y / (n_y as f64);

    fill_xy_col(n, n_x, n_y, dim_x, dim_y, a, b, x, y);

    let mut p = 0;

    for i in 0..n_y {
        for j in 0..n_x {
            x[p] -= l_x / 2.0;  // Why p??
            y[p] -= l_y / 2.0;

            p += 1;
        }
    }
}


pub fn fill_xy_col(N: usize, n_x: usize, n_y: usize, dim_x: f64, dim_y: f64, a: f64, b: f64, x: &mut [f64], y: &mut [f64]) {

    let mut p = 0;

    let l_x = dim_x / n_x as f64;
    let l_y = dim_y / n_y as f64;

    for i in 0..n_y {
        for j in 0..n_x {
            x[p] = a + j as f64 * l_x + l_x / 2.0;
            y[p] = b + i as f64 * l_y + l_y / 2.0;

            p += 1;
        }
    }
}


pub fn fxy(x: f64, y: f64, z: f64, k: Complex64, dim_x: f64, dim_y: f64) -> Complex64 {
    let x0 = 0.0;
    let y0 = dim_y / 2.0 + 0.15;
    let z0 = 0.0;

    let r0 = ((x - x0)*(x - x0) + (y - y0)*(y - y0) + (z - z0)*(z - z0)).sqrt();

    (Complex64::i() * k * r0).exp() / r0
}

pub fn G(x1: f64, y1: f64, z1: f64, x2: f64, y2: f64, z2: f64, k0: Complex64) -> Complex64 {
    let r = ((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2)).sqrt();
    (Complex64::i() * k0 * r).exp() / r
}


pub fn calculate_matrix_col(point: usize, n_x: usize, n_y: usize,
                            K: &mut Vec<Complex64>, dim_x: f64, dim_y: f64, a: f64, b: f64, N: usize, ip1: usize, ip2: usize,
                            A: &mut Vec<Vec<Complex64>>, k0: Complex64) {

    let mut flag = 0;

    let mut x = vec![0.0; N];
    let mut y = vec![0.0; N];

    fill_xy_pos(point, N, n_x, n_y, dim_x, dim_y, a, b, &mut x, &mut y);

    let mut xc = vec![0.0; N];
    let mut yc = vec![0.0; N];

    fill_xy_col(N, n_x, n_y, dim_x, dim_y, a, b, &mut xc, &mut yc);

    for i in 0..N {
        for j in 0..N {
            flag = 0;
            if i == j {
                A[i][j] = 1.0 / K[i];
                flag = 1;
            }
            A[i][j] -= integral_col(flag, n_x, n_y, dim_x, dim_y, a, b, ip1, x[j], y[j], xc[i], yc[i], k0);
        }
    }
}


pub fn rpart_col(N: usize,  n_x: usize, n_y: usize, dim_x: f64, dim_y: f64, a: f64, b: f64,
                 k0: Complex64, ip1: usize, ip2: usize, B: &mut Vec<Complex64>) {

    let mut x= vec![0.0; N];
    let mut y= vec![0.0; N];

    fill_xy_col(N, n_x, n_y, dim_x, dim_y, a, b, &mut x, &mut y);

    for i in 0..N {
        B[i] = fxy(x[i], y[i], 0.0, k0, dim_x, dim_y);
    }

}


pub fn integral_col(flag: usize, num_x: usize, num_y: usize, dim_x: f64, dim_y: f64, a: f64, b: f64, ip1: usize, xx1: f64, xx2: f64, xvv: f64, yvv: f64, k0: Complex64) -> Complex64 {

    let mut d: usize;

    let mut ret: Complex64 = Complex64::zero();
    let mut s: Complex64 = Complex64::zero();
    let mut x1: f64;
    let mut x2: f64;
    let mut x3: f64;
    let mut hx1: f64;
    let mut hy1: f64;
    let mut len_x: f64;
    let mut len_y: f64;

    // Calculate length in x and y direction
    len_x = dim_x / (num_x as f64);
    len_y = dim_y / (num_y as f64);

    d = 1;
    ret = Complex64::zero();

    if flag == 0 {
        hx1 = len_x / (ip1 as f64);
        hy1 = len_y / (ip1 as f64);

        s = Complex64::new(0.0, 0.0);
        for i1 in 0..ip1 {
            x1 = xx1 + hx1 / 2.0 + i1 as f64 * hx1;
            for i2 in 0..ip1 {
                x2 = xx2 + hy1 / 2.0 + i2 as f64 * hy1;

                s += G(x1, x2, 0.0, xvv, yvv, 0.0, k0);
            }
        }

        ret = s * hx1 * hy1;

    } else {
        hx1 = len_x / ((ip1+d) as f64);
        hy1 = len_y / ((ip1+d) as f64);

        s = Complex64::zero();
        for i1 in 0..ip1+d {
            x1 = xx1 + hx1 / 2.0 + i1 as f64 * hx1;
            for i2 in 0..ip1+d {
                x2 = xx2 + hy1 / 2.0 + i2 as f64 * hy1;

                s += G(x1, x2, 0.0, xvv, yvv, 0.0, k0);
            }
        }

        ret = s * hx1 * hy1;
    }

    ret
}
