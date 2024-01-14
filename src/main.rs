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
use std::process::{Command, exit, Output};
use std::time::Instant;
use num::complex::{Complex64, ComplexFloat};
use num::{Complex, One, Zero};
use consts::*;
use gcg2d::solvers::{solve, TaskParams};


fn main() {

    // ---------------------------------------------------------------------------------------------------------------

    let task = TaskParams {
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
        neuro_use: false,

        // Расчёт поля в точках наблюдения
        vych_calc: true,

        // Загрузка Uvych из файлов
        load_uvych_from_files: false,

        // Внесение шума в Uvych
        add_noise_uvych: false,
        pct_noise_uvych: 0.00000000001,

        // Решить обратную задачу
        solve_inverse: true,

        // Пути к файлам
        input_dir: PathBuf::from("./input"),
        output_dir: PathBuf::from("./output"),
    };

    solve(task);
}

