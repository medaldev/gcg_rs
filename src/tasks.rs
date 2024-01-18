use std::f64::consts::PI;
use std::path::{Path, PathBuf};
use num::complex::Complex64;
use crate::consts::{EWAVE, GIGA};


pub struct TaskParameters {
    pub p: usize,
    pub point: usize,
    pub n_x: usize,
    pub n_y: usize,
    pub n: usize,
    pub ip1: usize,
    pub ip2: usize,
    pub hz: f64,
    pub k0: Complex64,
    pub dim_x: f64,
    pub dim_y: f64,
    pub a: f64,
    pub b: f64,
    pub shift: f64,
    pub alpha: f64,
    pub k1: Complex64,
    pub model: PathBuf,
}

impl Default for TaskParameters {
    fn default() -> Self {
        TaskParameters::base(30, 2, 3, 3)
    }
}

impl TaskParameters {

    pub fn base(p: usize, point: usize, ip1: usize, ip2: usize) -> Self {
        TaskParameters::init(p, point, ip1, ip2, 1.1 * GIGA, Complex64::new(0.4, 0.0), 0.15, 0.15, 0.01, "./models/model_12.pt")

    }
    pub fn init(p: usize, point: usize, ip1: usize, ip2: usize, hz: f64, k1: Complex64, dim_x: f64, dim_y: f64, alpha: f64, model: &str) -> Self {
        let n_x = p * point;
        let n_y = p * point;

        TaskParameters {
            p,
            point,
            n_x,
            n_y,
            n: n_x * n_y,
            ip1,
            ip2,
            hz,
            k0: Complex64::new(2.0 * PI * hz / EWAVE, 0.0),
            dim_x,
            dim_y,
            a: -dim_x / 2.0,
            b: -dim_y / 2.0,
            shift: dim_x / 2.0,
            alpha,
            k1,
            model: PathBuf::from(model),
        }
    }

    pub fn set_hz(&mut self, hz: f64) {
        self.hz = hz
    }
}


pub struct SolutionSettings {
    pub use_initial_k: bool,
    pub load_init_k_w_from_files: bool,
    pub solve_direct: bool,
    pub load_j_from_files: bool,
    pub add_noise_j: bool,
    pub pct_noise_j: f64,
    pub neuro_use_j: bool,
    pub neuro_use_k_inv: bool,
    pub vych_calc: bool,
    pub load_uvych_from_files: bool,
    pub add_noise_uvych: bool,
    pub pct_noise_uvych: f64,
    pub solve_inverse: bool,
    pub input_dir: PathBuf,
    pub output_dir: PathBuf,
}



pub fn init_data_and_full_cycle<P: AsRef<Path>>(inp_dir: P, out_dir: P) -> SolutionSettings where PathBuf: From<P> {
    SolutionSettings {
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
        input_dir: PathBuf::from(inp_dir),
        output_dir: PathBuf::from(out_dir),
        neuro_use_k_inv: false,
    }
}

pub fn init_data_and_full_cycle_with_denoise_J<P: AsRef<Path>>(inp_dir: P, out_dir: P, noise_j_pct: f64, use_nn_j: bool) -> SolutionSettings where PathBuf: From<P> {
    SolutionSettings {
        // Задание начальных значений K
        use_initial_k: true,

        // Загрузка K из файлов
        load_init_k_w_from_files: false,

        // Решить прямую задачу
        solve_direct: true,

        // Загрузка J из файлов
        load_j_from_files: false,

        // Внесение шума в J
        add_noise_j: true,
        pct_noise_j: noise_j_pct,

        // Использовать нейросеть для очистки J
        neuro_use_j: use_nn_j,
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
        input_dir: PathBuf::from(inp_dir),
        output_dir: PathBuf::from(out_dir),
    }
}

pub fn init_data_and_full_cycle_with_denoise_k_inv<P: AsRef<Path>>(inp_dir: P, out_dir: P, noise_uvych_pct: f64) -> SolutionSettings where PathBuf: From<P> {
    SolutionSettings {
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
        neuro_use_k_inv: true,

        // Расчёт поля в точках наблюдения
        vych_calc: true,

        // Загрузка Uvych из файлов
        load_uvych_from_files: false,

        // Внесение шума в Uvych
        add_noise_uvych: true,
        pct_noise_uvych: noise_uvych_pct,

        // Решить обратную задачу
        solve_inverse: true,

        // Пути к файлам
        input_dir: PathBuf::from(inp_dir),
        output_dir: PathBuf::from(out_dir),
    }
}

pub fn only_from_saved_uvych<P: AsRef<Path>>(inp_dir: P, out_dir: P) -> SolutionSettings where PathBuf: From<P> {
    SolutionSettings {
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
        neuro_use_j: false,
        neuro_use_k_inv: false,

        // Расчёт поля в точках наблюдения
        vych_calc: false,

        // Загрузка Uvych из файлов
        load_uvych_from_files: true,

        // Внесение шума в Uvych
        add_noise_uvych: false,
        pct_noise_uvych: 1e-6,

        // Решить обратную задачу
        solve_inverse: true,

        // Пути к файлам
        input_dir: PathBuf::from(inp_dir),
        output_dir: PathBuf::from(out_dir),
    }
}

pub fn load_k_w_and_full_cycle<P: AsRef<Path>>(inp_dir: P, out_dir: P) -> SolutionSettings where PathBuf: From<P> {
    SolutionSettings {
        // Задание начальных значений K
        use_initial_k: false,

        // Загрузка K из файлов
        load_init_k_w_from_files: true,

        // Решить прямую задачу
        solve_direct: true,

        // Загрузка J из файлов
        load_j_from_files: false,

        // Внесение шума в J
        add_noise_j: false,
        pct_noise_j: 0.5,

        // Использовать нейросеть для очистки J
        neuro_use_j: false,
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
        input_dir: PathBuf::from(inp_dir),
        output_dir: PathBuf::from(out_dir),
    }
}