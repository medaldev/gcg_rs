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

mod tasks;

use std::path::PathBuf;
use gcg2d::solvers::{solve};
use gcg2d::tasks::*;


fn main() -> anyhow::Result<()> {

    // ---------------------------------------------------------------------------------------------------------------



    //let task = init_data_and_full_cycle_with_denoise_k_inv("", "./output", 1e-10);
    //let task = load_k_w_and_full_cycle(dir_1, "./output");
    //let task = only_from_saved_uvych("./input", "./output/trash");

    //let task = only_from_saved_uvych("./output/trash", "./output");
    //let settings = init_data_and_full_cycle_with_denoise_J("./output/trash", "./output", 0.8, true);
    //let task = init_data_and_full_cycle("asdf", "./output/trash");

    let settings = SolutionSettings {
        // Задание начальных значений K
        use_initial_k: false,

        // Загрузка K из файлов
        load_init_k_w_from_files: false,

        // Решить прямую задачу
        solve_direct: false,

        // Загрузка J из файлов
        load_j_from_files: true,

        // Внесение шума в J
        add_noise_j: true,
        pct_noise_j: 0.65,

        // Использовать нейросеть для очистки J
        neuro_use_j: true,
        neuro_use_k_inv: false,

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
        input_dir: PathBuf::from("./output"),
        output_dir: PathBuf::from("./output"),
    };

    let mut params = TaskParameters::from_grid(32, 2);
    solve(&settings, &mut params);

    Ok(())

}




