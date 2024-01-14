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
mod vych;
mod common;

use std::path::{Path, PathBuf};
use std::process::{Command, exit};
use std::time::Instant;
use num::complex::{Complex64, ComplexFloat};
use num::{Complex, One, Zero};
use rand::distributions::{Distribution, Uniform};
use consts::*;
use memory::*;
use matrix_system::{fill_xy_col};
use crate::common::{build_complex_vector, separate_re_im};
use crate::direct_problem::direct_problem;
use crate::initial::initial_k0;
use crate::inverse_problem::inverse_p;
use crate::linalg::{build_matrix, gauss, min_max_f64_vec};
use crate::matrix_system::{calculate_matrix_col, fill_xy_pos, fill_xyv, fxy, integral_col, rpart_col};
use crate::stream::{csv_complex, read_complex_vec_from_binary, read_complex_vector, read_long_vector, read_vec_from_binary_file, write_complex_vec_to_binary, write_complex_vector, write_complex_vector_r_i};


fn main() {

    // Задание начальных значений K
    let use_initial_k = true;

    // Загрузка K из файлов
    let load_init_k_w_from_files = false;

    // Решить прямую задачу
    let solve_direct = false;

    // Загрузка J из файлов
    let load_j_from_files = false;

    // Внесение шума в J
    let add_noise_j = false;
    let pct_noise = 0.5;

    // Использовать нейросеть для очистки J
    let neuro_use = false;

    // Расчёт поля в точках наблюдения
    let vych_calc = false;

    // Загрузка Uvych из файлов
    let load_uvych_from_files = true;

    // Решить обратную задачу
    let solve_inverse = true;

    // Пути к файлам
    let input_dir = Path::new("./input");
    let output_dir = Path::new("./output");

    // ---------------------------------------------------------------------------------------------------------------

    let global_start_time = Instant::now();

    println!("\n******************************START***************************************");

    println!("N = {}", N);

    println!("lambda = {} {}", (2.0*PI / K0).abs(), DIM_X / N_X as f64);
    println!("K0 = {:?}", K0);
    println!("Frequency = {:?}", HZ);

    let mut J = create_vector_memory(N, Complex64::new(0.1, 0.0));
    let mut K = create_vector_memory(N, K1);
    let mut W = create_vector_memory(N, Complex64::new(1.0, 0.0));
    let mut Bvych = create_vector_memory(N, Complex64::zero());
    let mut Uvych = create_vector_memory(N, Complex64::zero());
    let mut K_inv = create_vector_memory(N, Complex64::zero());
    let mut J_inv = create_vector_memory(N, Complex64::zero());


    if use_initial_k {
        // Задание начальных значений K
        initial_k0(N, &mut K, &mut W);
        println!("K was initialised from initial_k0 function.");
        write_complex_vector(&W, output_dir.join("WW.xls"), N_X, N_Y);
    }

    if load_init_k_w_from_files {
        // Загрузка K и W из xls файлов
        read_complex_vector(&mut K, input_dir.join("K_r.xls"), input_dir.join("K_i.xls"), N_X, N_Y);
        read_complex_vector(&mut W, input_dir.join("W_r.xls"), input_dir.join("W_i.xls"), N_X, N_Y);
        println!("K and W were loaded from external files.");
    }

    // Вывод начальных значений K
    write_complex_vector(&K, output_dir.join("K.xls"), N_X, N_Y);
    write_complex_vector_r_i(&K, output_dir.join("K_r.xls"), output_dir.join("K_i.xls"), N_X, N_Y);
    csv_complex(output_dir.join("K_init_r.csv"), output_dir.join("K_init_i.csv"),  N, N_X, N_Y, DIM_X, DIM_Y, A, B, &K);

    // ----------------------------------------------------------------------------------------------------------------
    if solve_direct {
        // Решение прямой задачи
        let mut start_time = Instant::now();
        println!("\n******************************DIRECT_PROBLEM******************************");
        direct_problem(N, N_X, N_Y, IP1, IP2, &mut W, &mut K, &mut J);

        println!(">> Direct problem time: {:?} seconds", start_time.elapsed().as_secs());
    }

    // ----------------------------------------------------------------------------------------------------------------
    if load_j_from_files {
        // Загрузка J из xls файлов

        // let j_re = read_long_vector(input_dir.join("J_re.xls");
        // let j_im = read_long_vector(input_dir.join("J_im.xls");
        // let j_loaded = build_complex_vector(N, j_re, j_im);
        // assert_eq!(J.len(), j_loaded.len());
        //J = j_loaded;

        read_complex_vector(&mut J, input_dir.join("J_re.xls"), input_dir.join("J_im.xls"), N_X, N_Y);

        println!("J was loaded from external files.");
    }


    // ----------------------------------------------------------------------------------------------------------------
    {
        // Сохранение результатов решения прямой задачи
        csv_complex(output_dir.join("K_dir_r.csv"), output_dir.join("K_dir_i.csv"),  N, N_X, N_Y, DIM_X, DIM_Y, A, B, &K);
        csv_complex(output_dir.join("J_dir_r.csv"), output_dir.join("J_dir_i.csv"),  N, N_X, N_Y, DIM_X, DIM_Y, A, B, &J);
        //write_complex_vector(&J, output_dir.join("J_dir_r.xls",output_dir.join("J_dir_i.xls", NUM_X, NUM_Y, N_X, N_Y);
        write_complex_vector_r_i(&J,output_dir.join("J_dir_r.xls") ,output_dir.join("J_dir_i.xls"), N_X, N_Y);
    }

    // ----------------------------------------------------------------------------------------------------------------
    if add_noise_j {
        // Внесение шума в J
        add_noise(&mut J, pct_noise);
        println!("Noise {} added to J.", pct_noise);

        // Запись зашумлённых данных
        csv_complex(output_dir.join("J_dir_r_noised.csv"), output_dir.join("J_dir_i_noised.csv"),  N, N_X, N_Y, DIM_X, DIM_Y, A, B, &J);
        //write_complex_vector(&J, output_dir.join("J_dir_r.xls",output_dir.join("J_dir_i.xls", NUM_X, NUM_Y, N_X, N_Y);
        write_complex_vector_r_i(&J, output_dir.join("J_dir_r_noised.xls") ,output_dir.join("J_dir_i_noised.xls"), N_X, N_Y);
    }

    // ----------------------------------------------------------------------------------------------------------------
    if neuro_use {
        // Подавление шума с помощью нейронной сети
        let start_time = Instant::now();

        println!("Using neural network model to denoise J.");

        {
            let j2_denoised_re = neuro::run(output_dir.join("J_dir_r.xls"), PathBuf::from(MODEL),
                                            output_dir.join("J_dir_r_denoised.xls"), true).unwrap().concat();
            let j2_denoised_im = neuro::run(output_dir.join("J_dir_i.xls"), PathBuf::from(MODEL),
                                            output_dir.join("J_dir_i_denoised.xls"), true).unwrap().concat();
            // let J2 = build_complex_vector(N, j2_denoised_re, j2_denoised_im);
        }


        read_complex_vector(&mut J, output_dir.join("J_dir_r_denoised.xls"), output_dir.join("J_dir_i_denoised.xls"), N_X, N_Y);

        //----------------------------------------------------------------------------------------------------------------

        let errors: Vec<f64> = J.iter().zip(J.iter()).map(|(j1, j2)| ((j1 - j2).abs() * 100.).round() / 100.).collect();
        println!("Avg error: {:?}", errors.iter().sum::<f64>() / errors.len() as f64);
        println!(">> Model inference time: {:?} seconds", start_time.elapsed().as_secs());

    }

    // ----------------------------------------------------------------------------------------------------------------
    if vych_calc {
        // Расчёт поля в точках наблюдения
        vych::r_part_vych(POINT, SHIFT, N_X, N_Y, DIM_X, DIM_Y, A, B, K0, N, IP1, &mut Bvych);
        vych::get_uvych(N, IP1, DIM_X, DIM_Y, &J, &mut Uvych, &Bvych, K0);
        println!("Uvych was calculated.");
    }

    // Загрузка Uvych из файлов
    if load_uvych_from_files {

        let Uvych_readed = read_complex_vec_from_binary(input_dir.join("Uvych_r"), input_dir.join("Uvych_i"));
        assert_eq!(Uvych.len(), Uvych_readed.len());
        Uvych = Uvych_readed;

        println!("Uvych was loaded from external files.");
    }

    {
        // Сохранение Uvych
        write_complex_vector_r_i(&Uvych, output_dir.join("Uvych_r.xls") ,output_dir.join("Uvych_i.xls") , N_X, N_Y);
        write_complex_vec_to_binary(&Uvych, output_dir.join("Uvych_r") , output_dir.join("Uvych_i"));
    }

    // ----------------------------------------------------------------------------------------------------------------
    if solve_inverse {
        // Обратная задача
        let start_time = Instant::now();
        inverse_p(&Uvych, &mut J_inv, &mut K_inv);
        println!(">> Inverse problem time: {:?} seconds", start_time.elapsed().as_secs());

        // ----------------------------------------------------------------------------------------------------------------
        write_complex_vector(&K_inv, output_dir.join("K_inv.xls"), N_X, N_Y);

        write_complex_vector_r_i(&K_inv, output_dir.join("K_inv_r.xls") ,output_dir.join("K_inv_i.xls"), N_X, N_Y);

        csv_complex(output_dir.join("K_inv_r.csv"), output_dir.join("K_inv_i.csv"),  N, N_X, N_Y, DIM_X, DIM_Y, A, B, &K_inv);
        csv_complex(output_dir.join("J_inv_r.csv"), output_dir.join("J_inv_i.csv"),  N, N_X, N_Y, DIM_X, DIM_Y, A, B, &J_inv);

    }

    let res_duration =  global_start_time.elapsed().as_secs();
    let eps_rnd = 1.0e2;
    println!("\n==========================================================================");
    println!(">> Total time: {:?} seconds ({:?} minutes)\n", res_duration, ((res_duration as f64) / 60.0 * eps_rnd).round() / eps_rnd);
}

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
