use std::fs::File;
use std::path::Path;
use num::complex::{Complex64, ComplexFloat};
use std::io::Write;
use crate::matrix_system::fill_xy_col;
use crate::memory::create_vector_memory;

pub fn write_complex_vector<P: AsRef<Path>>(U: &Vec<Complex64>, path1: P, path2: P, num_x: usize, num_y: usize, n_x: usize, n_y: usize) {
    let mut f1 = File::create(path1).unwrap();
    let mut f2 = File::create(path2).unwrap();

    let mut p = 0;

    for _i2 in 0..num_y {
        for _i1 in 0..num_x {
            write!(&mut f1, "{}\t", U[p].abs()).unwrap();
            p += 1;
        }
        writeln!(&mut f1).unwrap();
    }

    for _i2 in 0..n_y {
        for _i1 in 0..n_x {
            write!(&mut f2, "{}\t", U[p].abs()).unwrap();
            p += 1;
        }
        writeln!(&mut f2).unwrap();
    }

}


pub fn csv_complex<P: AsRef<Path>>(path_r: P, path_i: P, n: usize, num_x: usize, num_y: usize, n_x: usize, n_y: usize,
                                   dim_x: f64, dim_y: f64, a: f64, b: f64, KK: &Vec<Complex64>) {
    let n1 = num_x * num_y;

    let len_x = dim_x / n_x as f64;
    let len_y = dim_y / n_y as f64;

    let mut x = create_vector_memory(n, 0.0);
    let mut y = create_vector_memory(n, 0.0);

    fill_xy_col(n, num_x, num_y, n_x, n_y, dim_x, dim_y, a, b, &mut x, &mut y);

    let mut f1 = File::create(path_r).unwrap();
    let mut f2 = File::create(path_i).unwrap();

    //writeln!(&mut f1, "\"\"\,\"\, ").unwrap();

    writeln!(&mut f1, "\"\",\"\",\"\"").unwrap();
    for i1 in 0..n_x * n_y {
        writeln!(&mut f1, "{},{},{}", x[n1 + i1], y[n1 + i1], KK[n1 + i1].abs()).unwrap();
    }

    writeln!(&mut f2, "\"\",\"\",\"\"").unwrap();
    for i1 in 0..n_x * n_y {
        writeln!(&mut f2, "{},{},{}", x[n1 + i1], y[n1 + i1], KK[n1 + i1].im()).unwrap();
    }

}