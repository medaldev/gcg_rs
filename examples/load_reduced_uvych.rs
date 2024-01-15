use std::path::PathBuf;
use gcg2d::solvers::{solve, TaskParams};


fn main() {
    /*

        cargo run --example reduce_binary ./experiments/output50_2/Uvych_r  ./experiments/output50_2/Uvych_i ./input/Uvych_r ./input/Uvych_i 2

    */

    let mut task = TaskParams {
        // Задание начальных значений K
        use_initial_k: false,

        // Загрузка K из файлов
        load_init_k_w_from_files: false,

        // Решить прямую задачу
        solve_direct: false,

        // Загрузка J из файлов
        load_j_from_files: false,

        // Внесение шума в J
        add_noise_j: false,
        pct_noise_j: 0.5,

        // Использовать нейросеть для очистки J
        neuro_use: false,

        // Расчёт поля в точках наблюдения
        vych_calc: false,

        // Загрузка Uvych из файлов
        load_uvych_from_files: true,

        // Внесение шума в Uvych
        add_noise_uvych: false,
        pct_noise_uvych: 0.00000000001,

        // Решить обратную задачу
        solve_inverse: true,

        // Пути к файлам
        input_dir: PathBuf::from("./input"),
        output_dir: PathBuf::from("./output"),
    };

    solve(&task);
}
