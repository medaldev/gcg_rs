use std::fs::File;
use std::path::Path;
use num::complex::{Complex64, ComplexFloat};
use std::io::{BufRead, BufReader, Read, Write};
use crate::matrix_system::fill_xy_col;
use crate::memory::create_vector_memory;

use byteorder::{LittleEndian, WriteBytesExt, ReadBytesExt};
use crate::common::{build_complex_vector, separate_re_im};

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


// fn write_complex_matrix(file: &mut File, U: &Vec<Complex64>, cols: usize, rows: usize,  p: usize) {
//     for _i2 in 0..rows {
//         for _i1 in 0..cols {
//             write!(&mut file, "{}\t", U[p].re).unwrap();
//             p += 1;
//         }
//         writeln!(&mut f1_r).unwrap();
//     }
// }

pub fn write_complex_vector_r_i<P: AsRef<Path>>(U: &Vec<Complex64>, path1_r: P, path1_i: P, path2_r: P, path2_i: P,
                                                num_x: usize, num_y: usize, n_x: usize, n_y: usize) {
    let mut f1_r = File::create(path1_r).unwrap();
    let mut f1_i = File::create(path1_i).unwrap();
    let mut f2_r = File::create(path2_r).unwrap();
    let mut f2_i = File::create(path2_i).unwrap();

    let mut p = 0;

    for _i2 in 0..num_y {
        for _i1 in 0..num_x {
            write!(&mut f1_r, "{}\t", U[p].re).unwrap();
            write!(&mut f1_i, "{}\t", U[p].im).unwrap();
            p += 1;
        }
        writeln!(&mut f1_r).unwrap();
        writeln!(&mut f1_i).unwrap();
    }

    for _i2 in 0..n_y {
        for _i1 in 0..n_x {
            write!(&mut f2_r, "{}\t", U[p].re).unwrap();
            write!(&mut f2_i, "{}\t", U[p].im).unwrap();
            p += 1;
        }
        writeln!(&mut f2_r).unwrap();
        writeln!(&mut f2_i).unwrap();
    }

}


pub fn read_long_vector<P: AsRef<Path>>(path: P) -> Vec<f64> {

    let mut vec = Vec::new();
    let f1_buf_r = BufReader::new(File::open(path).unwrap());

    for some_line in f1_buf_r.lines() {
        let line = some_line.unwrap();
        vec.push(str::parse::<f64>(line.as_str()).unwrap());
    }

    vec
}

pub fn write_complex_vec_to_binary<P: AsRef<Path>>(vec: &Vec<Complex64>, path_re: P, path_im: P) {
    let (re_part, im_part) = separate_re_im(vec);
    write_vec_to_binary_file(&re_part, path_re).unwrap();
    write_vec_to_binary_file(&im_part, path_im).unwrap();
}

pub fn read_complex_vec_from_binary<P: AsRef<Path>>(path_re: P, path_im: P) -> Vec<Complex64> {
    let re_part = read_vec_from_binary_file(path_re).unwrap();
    let im_part = read_vec_from_binary_file(path_im).unwrap();

    let n = re_part.len();
    assert_eq!(n, im_part.len());

    build_complex_vector(n, re_part, im_part)
}

pub fn write_vec_to_binary_file<P: AsRef<Path>>(vec: &Vec<f64>, filename: P) -> std::io::Result<()> {
    let mut file = File::create(filename)?;

    for &value in vec {
        file.write_f64::<LittleEndian>(value)?;
    }

    Ok(())
}

pub fn read_vec_from_binary_file<P: AsRef<Path>>(filename: P) -> std::io::Result<Vec<f64>> {
    let mut vec = Vec::new();
    let mut file = File::open(filename)?;
    let file_size = file.metadata()?.len();
    let num_elements = file_size / std::mem::size_of::<f64>() as u64;

    for _ in 0..num_elements {
        let value = file.read_f64::<LittleEndian>()?;
        vec.push(value);
    }

    Ok(vec)
}

// pub fn read_long_vector<P: AsRef<Path>>(U: &mut Vec<Complex64>, path1_r: P, path1_i: P, path2_r: P, path2_i: P,
//                                            num_x: usize, num_y: usize, n_x: usize, n_y: usize) {
//     let mut p = 0;
//
//     {
//         let f1_buf_r = BufReader::new(File::open(path1_r).unwrap());
//         let f1_buf_i = BufReader::new(File::open(path1_i).unwrap());
//
//
//         for (_i2, (line_r, line_i)) in (0..num_y).into_iter().zip(f1_buf_r.lines().zip(f1_buf_i.lines())) {
//             for (_i1, (val_str_r, val_str_i)) in (0..num_x).into_iter().zip(line_r.unwrap().split("\t").zip(line_i.unwrap().split("\t"))) {
//                 U[p] = Complex64::new(str::parse::<f64>(val_str_r).unwrap(), str::parse::<f64>(val_str_i).unwrap());
//                 p += 1;
//             }
//         }
//     }
//



// pub fn read_binary_file<P: AsRef<Path>>(path: P, n: usize) -> Vec<Complex64> {
//     let mut reader = BufReader::new(
//         File::open(path)
//             .expect("Failed to open file")
//     );
//
//     let mut vec = Vec::new();
//     let mut buffer = [0u8; std::mem::size_of::<f64>()];
//
//     loop {
//         reader.read_exact(&mut buffer).expect("Unexpected error during read");
//         let real = f64::from_le_bytes(buffer);
//
//         reader.read_exact(&mut buffer).expect("Unexpected error during read");
//         let imag = f64::from_le_bytes(buffer);
//
//         let complex = Complex64::new(real, imag);
//         vec.push(complex);
//     }
//     vec
//
// }

pub fn read_complex_vector<P: AsRef<Path>>(U: &mut Vec<Complex64>, path1_r: P, path1_i: P, path2_r: P, path2_i: P,
                                           num_x: usize, num_y: usize, n_x: usize, n_y: usize) {
    let mut p = 0;

    {
        let f1_buf_r = BufReader::new(File::open(path1_r).unwrap());
        let f1_buf_i = BufReader::new(File::open(path1_i).unwrap());


        for (_i2, (line_r, line_i)) in (0..num_y).into_iter().zip(f1_buf_r.lines().zip(f1_buf_i.lines())) {
            for (_i1, (val_str_r, val_str_i)) in (0..num_x).into_iter().zip(line_r.unwrap().split("\t").zip(line_i.unwrap().split("\t"))) {
                U[p] = Complex64::new(str::parse::<f64>(val_str_r).unwrap(), str::parse::<f64>(val_str_i).unwrap());
                p += 1;
            }
        }
    }

    {
        let f2_buf_r = BufReader::new(File::open(path2_r).unwrap());
        let f2_buf_i = BufReader::new(File::open(path2_i).unwrap());

        for (_i2, (line_r, line_i)) in (0..n_y).into_iter().zip(f2_buf_r.lines().zip(f2_buf_i.lines())) {
            for (_i1, (val_str_r, val_str_i)) in (0..n_x).into_iter().zip(line_r.unwrap().split("\t").zip(line_i.unwrap().split("\t"))) {
                U[p] = Complex64::new(str::parse::<f64>(val_str_r).unwrap(), str::parse::<f64>(val_str_i).unwrap());
                p += 1;
            }
        }
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


pub fn read_xls<P: AsRef<Path>>(path: P) -> Vec<Vec<f64>> {
    let f = BufReader::new(File::open(path).unwrap());

    // Parse the file into a 2D vector
    let arr: Vec<Vec<f64>> = f.lines()
        .map(|l| l.unwrap().split('\t')
            .map(|number| number.parse::<f64>().unwrap())
            .collect())
        .collect();
    arr
}

