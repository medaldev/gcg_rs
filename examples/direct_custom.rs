use std::path::PathBuf;
use clap::{arg, Parser};
use gcg2d::solvers::solve;
use gcg2d::tasks::{init_data_and_full_cycle, SolutionSettings, TaskParameters};

#[derive(Parser)]
struct Cli {
    #[clap(long = "path_from")]
    path_from: PathBuf,

    #[clap(long = "path_save")]
    path_save: PathBuf,

    #[clap(long = "p")]
    p: usize,

    #[clap(long = "point")]
    point: usize,

    #[clap(long = "ip1")]
    ip1: usize,

    #[clap(long = "ip2")]
    ip2: usize,
}

fn main() {

    let args = Cli::parse();

    let settings = SolutionSettings {
        // Задание начальных значений K
        use_initial_k: true,

        // Загрузка K из файлов
        load_init_k_w_from_files: false,

        // Решить прямую задачу
        solve_direct: true,

        // Загрузка J из файлов
        load_j_from_files: false,

        // Внесение шума в J
        add_noise_j: false,
        pct_noise_j: 0.5,

        // Использовать нейросеть для очистки J
        neuro_use_j: false,

        // Расчёт поля в точках наблюдения
        vych_calc: true,

        // Загрузка Uvych из файлов
        load_uvych_from_files: false,

        // Внесение шума в Uvych
        add_noise_uvych: false,
        pct_noise_uvych: 1e-6,

        // Решить обратную задачу
        solve_inverse: true,

        // Пути к файлам
        input_dir: args.path_from.clone(),
        output_dir: args.path_save.clone(),
        neuro_use_k_inv: false,
    };

    solve(&settings, &mut TaskParameters::base(args.p, args.point, args.ip1, args.ip2));
}