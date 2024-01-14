use std::f64::consts::PI;
use std::path::PathBuf;
use std::time::Instant;
use num::complex::{Complex64, ComplexFloat};
use num::Zero;
use crate::consts::{A, B, DIM_X, DIM_Y, HZ, IP1, IP2, K0, K1, MODEL, N, N_X, N_Y, POINT, SHIFT};
use crate::direct_problem::direct_problem;
use crate::initial::initial_k0;
use crate::memory::create_vector_memory;
use crate::{neuro, vych};
use crate::common::add_noise;
use crate::inverse_problem::inverse_p;
use crate::stream::{csv_complex, read_complex_vec_from_binary, read_complex_vector, write_complex_vec_to_binary, write_complex_vector, write_complex_vector_r_i};


pub struct TaskParams {
    pub use_initial_k: bool,
    pub load_init_k_w_from_files: bool,
    pub solve_direct: bool,
    pub load_j_from_files: bool,
    pub add_noise_j: bool,
    pub pct_noise_j: f64,
    pub neuro_use: bool,
    pub vych_calc: bool,
    pub load_uvych_from_files: bool,
    pub add_noise_uvych: bool,
    pub pct_noise_uvych: f64,
    pub solve_inverse: bool,
    pub input_dir: PathBuf,
    pub output_dir: PathBuf,
}

pub fn solve(p: TaskParams)

{
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


    if p.use_initial_k {
        // Задание начальных значений K
        initial_k0(N, &mut K, &mut W);
        println!("K was initialised from initial_k0 function.");
        write_complex_vector(&W, p.output_dir.join("WW.xls"), N_X, N_Y);
    }

    if p.load_init_k_w_from_files {
        // Загрузка K и W из xls файлов
        read_complex_vector(&mut K, p.input_dir.join("K_r.xls"), p.input_dir.join("K_i.xls"), N_X, N_Y);
        read_complex_vector(&mut W, p.input_dir.join("W_r.xls"), p.input_dir.join("W_i.xls"), N_X, N_Y);
        println!("K and W were loaded from external files.");
    }

    // Вывод начальных значений K
    write_complex_vector(&K, p.output_dir.join("K.xls"), N_X, N_Y);
    write_complex_vector_r_i(&K, p.output_dir.join("K_r.xls"), p.output_dir.join("K_i.xls"), N_X, N_Y);
    csv_complex(p.output_dir.join("K_init_r.csv"), p.output_dir.join("K_init_i.csv"),  N, N_X, N_Y, DIM_X, DIM_Y, A, B, &K);

    // ----------------------------------------------------------------------------------------------------------------
    if p.solve_direct {
        // Решение прямой задачи
        let mut start_time = Instant::now();
        println!("\n******************************DIRECT_PROBLEM******************************");
        direct_problem(N, N_X, N_Y, IP1, IP2, &mut W, &mut K, &mut J);

        println!(">> Direct problem time: {:?} seconds", start_time.elapsed().as_secs());
    }

    // ----------------------------------------------------------------------------------------------------------------
    if p.load_j_from_files {
        // Загрузка J из xls файлов

        // let j_re = read_long_vector(input_dir.join("J_re.xls");
        // let j_im = read_long_vector(input_dir.join("J_im.xls");
        // let j_loaded = build_complex_vector(N, j_re, j_im);
        // assert_eq!(J.len(), j_loaded.len());
        //J = j_loaded;

        read_complex_vector(&mut J, p.input_dir.join("J_re.xls"), p.input_dir.join("J_im.xls"), N_X, N_Y);

        println!("J was loaded from external files.");
    }


    // ----------------------------------------------------------------------------------------------------------------
    {
        // Сохранение результатов решения прямой задачи
        csv_complex(p.output_dir.join("K_dir_r.csv"), p.output_dir.join("K_dir_i.csv"),  N, N_X, N_Y, DIM_X, DIM_Y, A, B, &K);
        csv_complex(p.output_dir.join("J_dir_r.csv"), p.output_dir.join("J_dir_i.csv"),  N, N_X, N_Y, DIM_X, DIM_Y, A, B, &J);
        //write_complex_vector(&J, p.output_dir.join("J_dir_r.xls",p.output_dir.join("J_dir_i.xls", NUM_X, NUM_Y, N_X, N_Y);
        write_complex_vector_r_i(&J,p.output_dir.join("J_dir_r.xls") ,p.output_dir.join("J_dir_i.xls"), N_X, N_Y);
    }

    // ----------------------------------------------------------------------------------------------------------------
    if p.add_noise_j {
        // Внесение шума в J
        add_noise(&mut J, p.pct_noise_j);
        println!("Noise {} added to J.", p.pct_noise_j);

        // Запись зашумлённых данных
        csv_complex(p.output_dir.join("J_dir_r_noised.csv"), p.output_dir.join("J_dir_i_noised.csv"),  N, N_X, N_Y, DIM_X, DIM_Y, A, B, &J);
        //write_complex_vector(&J, p.output_dir.join("J_dir_r.xls",p.output_dir.join("J_dir_i.xls", NUM_X, NUM_Y, N_X, N_Y);
        write_complex_vector_r_i(&J, p.output_dir.join("J_dir_r_noised.xls") ,p.output_dir.join("J_dir_i_noised.xls"), N_X, N_Y);
    }

    // ----------------------------------------------------------------------------------------------------------------
    if p.neuro_use {
        // Подавление шума с помощью нейронной сети
        let start_time = Instant::now();

        println!("Using neural network model to denoise J.");

        {
            let j2_denoised_re = neuro::run(p.output_dir.join("J_dir_r.xls"), PathBuf::from(MODEL),
                                            p.output_dir.join("J_dir_r_denoised.xls"), true).unwrap().concat();
            let j2_denoised_im = neuro::run(p.output_dir.join("J_dir_i.xls"), PathBuf::from(MODEL),
                                            p.output_dir.join("J_dir_i_denoised.xls"), true).unwrap().concat();
            // let J2 = build_complex_vector(N, j2_denoised_re, j2_denoised_im);
        }


        read_complex_vector(&mut J, p.output_dir.join("J_dir_r_denoised.xls"), p.output_dir.join("J_dir_i_denoised.xls"), N_X, N_Y);

        //----------------------------------------------------------------------------------------------------------------

        let errors: Vec<f64> = J.iter().zip(J.iter()).map(|(j1, j2)| ((j1 - j2).abs() * 100.).round() / 100.).collect();
        println!("Avg error: {:?}", errors.iter().sum::<f64>() / errors.len() as f64);
        println!(">> Model inference time: {:?} seconds", start_time.elapsed().as_secs());

    }

    // ----------------------------------------------------------------------------------------------------------------
    if p.vych_calc {
        // Расчёт поля в точках наблюдения
        vych::r_part_vych(POINT, SHIFT, N_X, N_Y, DIM_X, DIM_Y, A, B, K0, N, IP1, &mut Bvych);
        vych::get_uvych(N, IP1, DIM_X, DIM_Y, &J, &mut Uvych, &Bvych, K0);
        println!("Uvych was calculated.");
    }

    // Загрузка Uvych из файлов
    if p.load_uvych_from_files {

        let Uvych_readed = read_complex_vec_from_binary(p.input_dir.join("Uvych_r"), p.input_dir.join("Uvych_i"));
        assert_eq!(Uvych.len(), Uvych_readed.len());
        Uvych = Uvych_readed;

        println!("Uvych was loaded from external files.");
    }

    if p.add_noise_uvych {
        // Внесение шума в Uvych
        add_noise(&mut Uvych, p.pct_noise_uvych);
        println!("Noise {} added to Uvych.", p.pct_noise_uvych);

        // Запись зашумлённых данных
        csv_complex(p.output_dir.join("J_dir_r_noised.csv"), p.output_dir.join("J_dir_i_noised.csv"),  N, N_X, N_Y, DIM_X, DIM_Y, A, B, &J);
        //write_complex_vector(&J, p.output_dir.join("J_dir_r.xls",p.output_dir.join("J_dir_i.xls", NUM_X, NUM_Y, N_X, N_Y);
        write_complex_vector_r_i(&J, p.output_dir.join("J_dir_r_noised.xls") ,p.output_dir.join("J_dir_i_noised.xls"), N_X, N_Y);
    }

    {
        // Сохранение Uvych
        write_complex_vector_r_i(&Uvych, p.output_dir.join("Uvych_r.xls") ,p.output_dir.join("Uvych_i.xls") , N_X, N_Y);
        write_complex_vec_to_binary(&Uvych, p.output_dir.join("Uvych_r") , p.output_dir.join("Uvych_i"));
    }

    // ----------------------------------------------------------------------------------------------------------------
    if p.solve_inverse {
        // Обратная задача
        let start_time = Instant::now();
        inverse_p(&Uvych, &mut J_inv, &mut K_inv);
        println!(">> Inverse problem time: {:?} seconds", start_time.elapsed().as_secs());

        // ----------------------------------------------------------------------------------------------------------------
        write_complex_vector(&K_inv, p.output_dir.join("K_inv.xls"), N_X, N_Y);

        write_complex_vector_r_i(&K_inv, p.output_dir.join("K_inv_r.xls") ,p.output_dir.join("K_inv_i.xls"), N_X, N_Y);

        csv_complex(p.output_dir.join("K_inv_r.csv"), p.output_dir.join("K_inv_i.csv"),  N, N_X, N_Y, DIM_X, DIM_Y, A, B, &K_inv);
        csv_complex(p.output_dir.join("J_inv_r.csv"), p.output_dir.join("J_inv_i.csv"),  N, N_X, N_Y, DIM_X, DIM_Y, A, B, &J_inv);

    }

    let res_duration =  global_start_time.elapsed().as_secs();
    let eps_rnd = 1.0e2;
    println!("\n==========================================================================");
    println!(">> Total time: {:?} seconds ({:?} minutes)\n", res_duration, ((res_duration as f64) / 60.0 * eps_rnd).round() / eps_rnd);
}
