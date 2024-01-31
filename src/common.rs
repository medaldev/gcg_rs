use std::fs;
use std::fs::File;
use std::path::Path;
use num::{Complex, Zero};
use num::complex::{Complex64, ComplexFloat};
use num::integer::Roots;
use rand::distributions::Uniform;
use rand::prelude::Distribution;
use crate::linalg::min_max_f64_vec;
use crate::memory::create_vector_memory;




pub fn add_noise(U: &mut Vec<Complex64>, pct: f64) {

    let mut rng = rand::thread_rng();
    let ppp = Uniform::from(0.0..pct);

    for num in U {
        *num *= 1. + ppp.sample(&mut rng);
    }
}


pub fn add_noise_re_im(U: &mut Vec<Complex64>, pct: f64) {

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

pub fn get_vec_im(U: &Vec<Complex64>) -> Vec<f64> {
    let n = U.len();
    let mut im = vec![0.0; n];
    for (i, num) in U.iter().enumerate() {
        im[i] = num.im;
    }
    im
}

pub fn get_vec_re(U: &Vec<Complex64>) -> Vec<f64> {
    let n = U.len();
    let mut re = vec![0.0; n];
    for (i, num) in U.iter().enumerate() {
        re[i] = num.re;
    }
    re
}

pub fn vec_to_matrix<T>(U: &Vec<T>, n_x: usize, n_y: usize) -> Vec<Vec<T>> where T: Zero + Clone + Copy {
    let mut res = vec![vec![T::zero(); n_x]; n_y];
    let mut p = 0;

    for i2 in 0..n_y {
        for i1 in 0..n_x {
            res[i2][i1] = U[p];
            p += 1;
        }
    }
    res
}

pub fn matrix_to_vec<T>(arr: Vec<Vec<T>>, n_x: usize, n_y: usize) -> Vec<T> where T: Zero + Clone + Copy {
    let mut U = vec![T::zero(); n_x * n_y];
    let mut p = 0;

    for i2 in 0..n_y {
        for i1 in 0..n_x {
            U[p] = arr[i2][i1];
            p += 1;
        }
    }
    U
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

pub fn reduce_matrix(n: usize, U: &Vec<Complex64>, start: usize, step: usize) -> (Vec<Complex64>, (usize, usize)) {
    let n = U.len().sqrt();
    let mut res = vec![];
    let mut p: usize = 0;
    for i in (start..n).step_by(step) {
        for j in (step..n).step_by(step) {
            res.push(U[i * n + j]);
            p += 1;
        }
    }
    let p_sq = p.sqrt();
    (res, (p_sq, p_sq))
}

pub fn avg_pool<T>(input_data: &[T], input_shape: (usize, usize),
                   kernel_shape: (usize, usize), stride: (usize, usize)
) -> (Vec<T>, (usize, usize))
    where T: Zero + std::ops::Div<f64> + Clone + std::ops::AddAssign<<T as std::ops::Div<f64>>::Output> + Copy
{
    let out_shape: (usize, usize) = get_conv2d_shape_out(input_shape, (0, 0), (1, 1), kernel_shape, stride);

    let (rows_out, cols_out) = out_shape;

    let mut res_data = vec![T::zero(); cols_out * rows_out];

    for (I_out, I) in (0..input_shape.0 - kernel_shape.0 + 1).step_by(stride.0).enumerate() {

        for (J_out, J) in (0..input_shape.1 - kernel_shape.1 + 1).step_by(stride.1).enumerate() {

            for (i_out, i) in (I..(I + kernel_shape.0)).enumerate() {
                for (j_out, j) in (J..(J + kernel_shape.1)).enumerate() {
                    res_data[I_out * cols_out + J_out] += input_data[i * input_shape.1 + j] / (kernel_shape.0 * kernel_shape.1) as f64;
                }
            }
        }
    }

    (res_data, out_shape)
}


pub fn get_conv2d_shape_out(
    h_in: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    kernel_size: (usize, usize),
    stride: (usize, usize))  -> (usize, usize)
{
    (
        (((h_in.0 + 2 * padding.0 - dilation.0 * (kernel_size.0 - 1) - 1) as f32 / stride.0 as f32) + 1.).floor() as usize,
        (((h_in.1 + 2 * padding.1 - dilation.1 * (kernel_size.1 - 1) - 1) as f32 / stride.1 as f32) + 1.).floor() as usize
    )
}

// pub fn reduce_matrix2(n: usize, U: &Vec<Complex64>, n_x: usize, n_y: usize, step: usize) -> Vec<Complex64> {
//     let n = U.len();
//
// }


pub fn copy_input_data(file_path: &Path, task_in_dir: &Path, task_out_dir: &Path, new_name: &str) {
    fs::create_dir_all(task_in_dir).unwrap();
    fs::create_dir_all(task_out_dir).unwrap();

    let dest = task_in_dir.join(Path::new(new_name).with_extension("xls"));
    fs::copy(file_path, dest.as_path()).unwrap();
}
