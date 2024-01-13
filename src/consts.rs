use num::complex::Complex64;
pub use std::f64::consts::PI;

pub const P: usize = 60;
pub const POINT: usize = 2;

pub const N_X: usize = P * POINT;
pub const N_Y: usize = P * POINT;

pub const N: usize = N_X * N_Y;

pub const IP1: usize = 3;
pub const IP2: usize = 3;

pub const KILO: f64 = 1e3;
pub const MEGA: f64 = 1e6;
pub const GIGA: f64 = 1e9;
pub const TERA: f64 = 1e12;

pub const HZ: f64 = 1.1 * GIGA;

pub const AWAVE: f64 = 340.29;
pub const EWAVE: f64 = 299792456.2;

pub const ERR: f64 = 0.000001;

pub const K0: Complex64 = Complex64::new(2.0 * PI * HZ / EWAVE, 0.0);

pub const DIM_X: f64 = 0.15;
pub const DIM_Y: f64 = 0.15;

pub const A: f64 = -DIM_X / 2.0;
pub const B: f64 = -DIM_Y / 2.0;

pub const SHIFT: f64 = DIM_X / 2.0;

pub const ALPHA: f64 = 0.01;

pub const K1: Complex64 = Complex64::new(0.4, 0.0);

pub const MODEL: &str = "./models/model_12.pt";
