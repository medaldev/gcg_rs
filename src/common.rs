use std::fs::File;
use std::path::Path;
use num::{Complex, Zero};
use num::complex::Complex64;
use num::integer::Roots;
use rand::distributions::Uniform;
use rand::prelude::Distribution;
use crate::linalg::min_max_f64_vec;
use crate::memory::create_vector_memory;



pub fn add_noise(U: &mut Vec<Complex64>, pct: f64) {
    let rngs = {
        let (re, im) = separate_re_im(&U);
        (min_max_f64_vec(&re), min_max_f64_vec(&im))
    };
    let max_noise_val_re = (rngs.0.1 - rngs.0.0) * pct;
    let max_noise_val_im = (rngs.1.1 - rngs.1.0) * pct;

    let mut rng = rand::thread_rng();
    let re_noise = Uniform::from(0.0..max_noise_val_re);
    let im_noise = Uniform::from(0.0..max_noise_val_im);

    for num in U {
        *num += Complex64::new(re_noise.sample(&mut rng), im_noise.sample(&mut rng));
    }
}


pub fn kerr(n: usize, alpha: f64, k1_: Complex64, U: &[Complex64], K: &mut [Complex64], W: &[Complex64]) {
    for p1 in 0..n {
        K[p1] = (k1_ + alpha *U[p1].norm()*U[p1]).norm()*W[p1];
    }
}

pub fn get_geometry(n: usize, W: &[Complex64]) {

}


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
