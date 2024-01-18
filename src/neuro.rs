use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use num::complex::Complex64;
use tch::{Device, IndexOp, Kind, Tensor};
use tch::utils::has_cuda;
use crate::common::{build_complex_vector, get_vec_im, get_vec_re, matrix_to_vec, vec_to_matrix};
use crate::linalg::min_max_f64_vec;
use crate::neuro;


pub fn run_from_file(data_path: &Path, model_path: &Path, save_res_xls: &Path, with_rescale: bool) -> anyhow::Result<Vec<Vec<f64>>> {

    let device = get_device();
    let kind = Kind::Float;

    let mut arr = xls_to_matrix(data_path);

    let (height, width) = (arr.len() as i64, arr[0].len() as i64);

    let source_rng = min_max_f64_vec(&arr.concat());
    if with_rescale {
        rescale_matrix(&mut arr, (0., 1.));
    }

    let tensor: Tensor = Tensor::from_slice(&arr.concat()).reshape(&[1, 1, height, width]).to_kind(kind).to_device(device);

    let mut model = tch::CModule::load(model_path)?;
    model.to(device, kind, true);

    let output = model.forward_ts(&[&tensor])?;

    let mut res = tensor_to_matrix(output, height, width);
    if with_rescale {
        rescale_matrix(&mut res, source_rng);
    }

    matrix_to_file(&res, save_res_xls).unwrap();

    Ok(res)
}

pub fn run_re(mut arr: Vec<Vec<f64>>, model_path: &Path, with_rescale: bool) -> anyhow::Result<Vec<Vec<f64>>> {

    let device = get_device();
    let kind = Kind::Float;

    let (height, width) = (arr.len() as i64, arr[0].len() as i64);

    let source_rng = min_max_f64_vec(&arr.concat());
    if with_rescale {
        rescale_matrix(&mut arr, (0., 1.));
    }

    let tensor: Tensor = Tensor::from_slice(&arr.concat()).reshape(&[1, 1, height, width]).to_kind(kind).to_device(device);

    let mut model = tch::CModule::load(model_path)?;
    model.to(device, kind, true);

    let output = model.forward_ts(&[&tensor])?;

    let mut res = tensor_to_matrix(output, height, width);
    if with_rescale {
        rescale_matrix(&mut res, source_rng);
    }

    Ok(res)
}

pub fn run_im(arr: &Vec<Complex64>, n: usize, n_x: usize, n_y: usize, model_path: &Path, with_rescale: bool) -> anyhow::Result<Vec<Complex64>> {
    let k_inv_denoised_r = matrix_to_vec(run_re(vec_to_matrix(&get_vec_re(&arr), n_x, n_y),
                                      model_path, with_rescale)?, n_x, n_y);
    let k_inv_denoised_i = matrix_to_vec(run_re(vec_to_matrix(&get_vec_im(&arr), n_x, n_y),
                                      model_path, with_rescale)?, n_x, n_y);

    let k_inv_denoised = build_complex_vector(n, k_inv_denoised_r, k_inv_denoised_i);
    Ok(k_inv_denoised)
}


pub fn rescale_val(val: f64, fr_rng: (f64, f64), to_rng: (f64, f64)) -> f64 {
    let delta1 = fr_rng.1 - fr_rng.0;
    let delta2 = to_rng.1 - to_rng.0;
    (delta2 * (val - fr_rng.0) / delta1) + to_rng.0
}

pub fn rescale_matrix(matrix: &mut Vec<Vec<f64>>, to_rng: (f64, f64)) {
    let fr_rng = {min_max_f64_vec(&matrix.concat())};
    for i in 0..matrix.len() {
        for j in 0..matrix[0].len() {
            matrix[i][j] = rescale_val(matrix[i][j], fr_rng, to_rng);
        }
    }
}



fn matrix_to_file<P: AsRef<Path>>(matrix: &Vec<Vec<f64>>, filename: P) -> std::io::Result<()> {
    let mut file = File::create(filename)?;
    let content = vec_to_string(matrix);
    file.write_all(content.as_bytes())
}

fn vec_to_string(matrix: &Vec<Vec<f64>>) -> String {
    matrix.iter()
        .map(|row| row.iter().map(|num| num.to_string()).collect::<Vec<String>>().join("\t"))
        .collect::<Vec<String>>().join("\n")
}

pub fn tensor_to_matrix(t: Tensor, height: i64, width: i64) -> Vec<Vec<f64>> {
    let mut out_arr = vec![vec![0.0; (width) as usize]; height as usize];
    for i in 0..height {
        for j in 0..width {
            out_arr[i as usize][j as usize] = t.f_double_value(&[0, 0, i, j]).unwrap();
        }
    }
    out_arr
}

pub fn get_device() -> Device {
    if has_cuda() {
        Device::Cuda(0)
    }
    else {
        Device::Cpu
    }
}




pub fn xls_to_matrix<P: AsRef<Path>>(path: P) -> Vec<Vec<f64>> {
    let f = BufReader::new(File::open(path).unwrap());

    // Parse the file into a 2D vector
    let arr: Vec<Vec<f64>> = f.lines()
        .map(|l| l.unwrap().split('\t')
            .filter(|str_part| str_part.len() > 0)
            .map(|number| number.parse::<f64>().unwrap())
            .collect())
        .collect();
    arr
}

