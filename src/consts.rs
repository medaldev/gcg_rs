use num::complex::Complex64;
pub use std::f64::consts::PI;

pub const p: usize = 10;
pub const point: usize = 2;

pub const NUM_X: usize = p;
pub const NUM_Y: usize = p;

pub const N_X: usize = NUM_X * point;
pub const N_Y: usize = NUM_Y * point;

pub const N1: usize = NUM_X * NUM_Y;
pub const N2: usize = N_X * N_Y;
pub const N: usize = N1 + N2;

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

pub const shift: f64 = DIM_X / 2.0;

pub const ALPHA: f64 = 0.01;

pub const K1: Complex64 = Complex64::new(0.4, 0.0);

pub const MODEL: &str = "./models/traced_model_9.pt";
