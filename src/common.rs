use std::fs::File;
use std::path::Path;
use num::{Complex, Zero};
use num::complex::Complex64;
use crate::memory::create_vector_memory;

pub fn separate_re_im(U: &Vec<Complex64>) -> (Vec<f64>, Vec<f64>){
    let n = U.len();
    let (mut re, mut im) = (vec![0.0; n], vec![0.0; n]);
    for (i, num) in U.iter().enumerate() {
        re[i] = num.re;
        im[i] = num.im;
    }
    (re, im)
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


pub fn separate_by_grids(U: &Vec<Complex64>, num_x: usize, num_y: usize, n_x: usize, n_y: usize) -> (Vec<Complex64>, Vec<Complex64>) {

    let (mut U1, mut U2) = (vec![], vec![]);

    let mut p = 0;

    for _i2 in 0..num_y {
        for _i1 in 0..num_x {
            U1.push(U[p]);
            p += 1;
        }
    }

    for _i2 in 0..n_y {
        for _i1 in 0..n_x {
            U2.push(U[p]);
            p += 1;
        }
    }

    (U1, U2)

}


pub fn union_grids(U1: &Vec<Complex64>, U2: &Vec<Complex64>, num_x: usize, num_y: usize, n_x: usize, n_y: usize) -> Vec<Complex64> {
    let mut U = vec![];

    for i in 0..num_y * num_x {
        U.push(U1[i]);
    }

    for i in 0..n_y * n_x {
        U.push(U2[i]);
    }

    U
}