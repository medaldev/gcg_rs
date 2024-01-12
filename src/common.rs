use std::fs::File;
use std::path::Path;
use num::{Complex, Zero};
use num::complex::Complex64;
use num::integer::Roots;
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

pub fn reduce_matrix(n: usize, U: &Vec<Complex64>, step: usize) -> Vec<Complex64> {
    let n = U.len().sqrt();
    let mut res = vec![];
    let mut p = 0;
    for i in (0..n).step_by(2) {
        for j in (0..n).step_by(2) {
            res.push(U[i * n + j]);
            p += 1;
        }
    }
    res
}


// pub fn reduce_matrix2(n: usize, U: &Vec<Complex64>, n_x: usize, n_y: usize, step: usize) -> Vec<Complex64> {
//     let n = U.len();
//
// }
