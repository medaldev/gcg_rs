use std::f64::consts::PI;
use std::path::{Path, PathBuf};
use std::time::Instant;
use num::complex::{Complex64, ComplexFloat};
use num::Zero;
use crate::direct_problem::direct_problem;
use crate::initial::initial_k0;
use crate::memory::create_vector_memory;
use crate::{neuro, vych};
use crate::common::{add_noise, add_noise_re_im, build_complex_vector, get_vec_im, get_vec_re, vec_to_matrix};
use crate::inverse_problem::inverse_p;
use crate::stream::{ComplexVectorSaver, csv_complex, read_complex_vec_from_binary, read_complex_vector, read_f64_from_file, read_value_from_binary_file, write_complex_vec_to_binary, write_complex_vector, write_complex_vector_r_i, write_f64_to_file, write_value_to_binary_file};
use crate::stream::SaveFormat::*;
use crate::tasks::{SolutionSettings, TaskParameters};


pub fn solve(settings: &SolutionSettings, params: &mut TaskParameters)

{
    let global_start_time = Instant::now();
    
    println!("\n******************************START***************************************");

    println!("n = {}", params.n);

    println!("lambda = {} {}", (2.0*PI / params.k0).abs(), params.dim_x / params.n_x as f64);
    println!("k0 = {:?}", params.k0);
    println!("Frequency = {:?}", params.hz);

    let mut J = create_vector_memory(params.n, Complex64::new(0.1, 0.0));
    let mut K = create_vector_memory(params.n, params.k1);
    let mut W = create_vector_memory(params.n, Complex64::new(1.0, 0.0));

    let vector_stream = ComplexVectorSaver::init(settings.input_dir.as_path(), settings.output_dir.as_path());

    if settings.use_initial_k {
        // Задание начальных значений K
        initial_k0(params.n, params.n_x, params.n_y, params.dim_x, params.dim_y, params.a, params.b, params.k1, &mut K, &mut W);
        println!("K was initialised from initial_k0 function.");
    }

    if settings.load_init_k_w_from_files {
        // Загрузка K и W из xls файлов
        params.k0 = Complex64::new(
            read_f64_from_file(settings.output_dir.join("k0_re.txt").as_path()).unwrap(),
            read_f64_from_file(settings.output_dir.join("k0_im.txt").as_path()).unwrap(),
        );
        vector_stream.load(&mut K, params.n, "K", Xls, params);
        vector_stream.load(&mut W, params.n, "W", Xls, params);

        println!("K and W were loaded from external files.");
    }

    // Вывод начальных значений K, W
    write_f64_to_file(params.k0.re, settings.output_dir.join("k0_re.txt").as_path()).unwrap();
    write_f64_to_file(params.k0.im, settings.output_dir.join("k0_im.txt").as_path()).unwrap();
    vector_stream.save(&K, "K", &[Xls, Csv], params);
    vector_stream.save(&W, "W", &[Xls, Csv], params);

    // ----------------------------------------------------------------------------------------------------------------
    if settings.solve_direct {
        // Решение прямой задачи
        let mut start_time = Instant::now();
        println!("\n******************************DIRECT_PROBLEM******************************");
        direct_problem(params.point, params.n, params.n_x, params.n_y, params.dim_x, params.dim_y, params.a, params.b, params.k0,
                       params.ip1, params.ip2, &mut W, &mut K, &mut J);

        println!(">> Direct problem time: {:?} seconds", start_time.elapsed().as_secs());
    }

    // ----------------------------------------------------------------------------------------------------------------
    if settings.load_j_from_files {

        vector_stream.load(&mut J, params.n, "J", Xls, params);

        println!("J was loaded from external files.");
    }

    vector_stream.save(&J, "J", &[Xls, Csv], params);


    // ----------------------------------------------------------------------------------------------------------------
    if settings.add_noise_j {
        // Внесение шума в J
        add_noise_re_im(&mut J, settings.pct_noise_j);
        println!("Noise {} added to J.", settings.pct_noise_j);

        // Запись зашумлённых данных
        vector_stream.save(&J, "J_noised", &[Xls, Csv], params);
    }

    if settings.neuro_use_j {
        // Подавление шума с помощью нейронной сети
        let start_time = Instant::now();

        println!("Using neural network model to denoise J.");

        let denoised = neuro::run_im(&J, params.n, params.n_x, params.n_y, params.model.as_path(), true).unwrap();
        vector_stream.save(&denoised, "J_denoised", &[Xls], params);
        J = denoised;

        println!(">> Model inference time: {:?} seconds", start_time.elapsed().as_secs());

    }

    // ----------------------------------------------------------------------------------------------------------------

    let mut Bvych = create_vector_memory(params.n, Complex64::zero());
    let mut Uvych = create_vector_memory(params.n, Complex64::zero());

    if settings.vych_calc {
        // Расчёт поля в точках наблюдения
        vych::r_part_vych(params.point, params.shift, params.n_x, params.n_y, params.dim_x, params.dim_y, params.a, params.b, params.k0, params.n, params.ip1, &mut Bvych);
        vych::get_uvych(params.point, params.n, params.n_x, params.n_y, params.ip1, params.dim_x, params.dim_y, params.a, params.b, params.shift,
                        &J, &mut Uvych, &Bvych, params.k0);
        println!("Uvych was calculated.");
    }

    // Загрузка Uvych из файлов
    if settings.load_uvych_from_files {

        vector_stream.load(&mut Uvych, params.n, "Uvych", Bin, params);

        println!("Uvych was loaded from external files.");
    }

    vector_stream.save(&Uvych, "Uvych", &[Xls, Bin], params);

    if settings.add_noise_uvych {
        // Внесение шума в Uvych
        add_noise(&mut Uvych, settings.pct_noise_uvych);
        println!("Noise {} added to Uvych.", settings.pct_noise_uvych);

        // Запись зашумлённых данных
        vector_stream.save(&Uvych, "Uvych_noised", &[Xls, Csv], params);
    }

    // ----------------------------------------------------------------------------------------------------------------
    if settings.solve_inverse {
        block_inverse(settings, params, &Uvych);
    }

    // Сохраняем параметры задачи
    params.save_to_file(settings.output_dir.join("params.json").as_path()).unwrap();

    let res_duration =  global_start_time.elapsed().as_secs();
    let eps_rnd = 1.0e2;
    println!("\n==========================================================================");
    println!(">> Total time: {:?} seconds ({:?} minutes)\n", res_duration, ((res_duration as f64) / 60.0 * eps_rnd).round() / eps_rnd);
}


pub fn block_inverse(settings: &SolutionSettings, params: &TaskParameters, Uvych: &Vec<Complex64>) {
    let mut K_inv = create_vector_memory(params.n, Complex64::zero());
    let mut J_inv = create_vector_memory(params.n, Complex64::zero());

    let vector_stream = ComplexVectorSaver::init(settings.input_dir.as_path(), settings.output_dir.as_path());

    // Обратная задача
    let start_time = Instant::now();
    inverse_p(params.n, params.n_x, params.n_y, params.point, params.shift, params.ip1, params.a, params.b, params.dim_x, params.dim_y, params.k0,
              &Uvych, &mut J_inv, &mut K_inv);
    println!(">> Inverse problem time: {:?} seconds", start_time.elapsed().as_secs());

    // ----------------------------------------------------------------------------------------------------------------
    vector_stream.save(&K_inv, "K_inv", &[Xls, Csv], params);
    vector_stream.save(&J_inv, "J_inv", &[Xls, Csv], params);

    if settings.neuro_use_k_inv {
        // Подавление шума с помощью нейронной сети
        let start_time = Instant::now();

        println!("Using neural network model to denoise K_inv.");

        let denoised = neuro::run_im(&K_inv, params.n, params.n_x, params.n_y, params.model.as_path(), true).unwrap();
        vector_stream.save(&denoised, "K_inv_denoised", &[Xls], params);

        println!(">> Model inference time: {:?} seconds", start_time.elapsed().as_secs());

    }
}

