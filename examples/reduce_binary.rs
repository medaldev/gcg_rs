use num::complex::Complex64;
extern crate gcg2d;

use std::env;
use num::integer::Roots;
use gcg2d::common::reduce_matrix;
use gcg2d::consts::{N_X, N_Y};
use gcg2d::stream::{read_complex_vec_from_binary, write_complex_vec_to_binary, write_complex_vector_r_i};

fn main() {
    let args: Vec<String> = env::args().collect();

    assert_eq!(args.len(), 6);
    println!("{:?}", args);

    let vector1 = read_complex_vec_from_binary(args[1].as_str(), args[2].as_str());

    println!("N_in = {:?}", vector1.len());

    let vector_res = reduce_matrix(vector1.len(), &vector1, str::parse::<usize>(args[5].as_str()).unwrap());
    println!("N_out = {:?}", vector_res.len());

    println!("sqrt = {:?}", vector_res.len().sqrt());

    write_complex_vec_to_binary(&vector_res, args[3].as_str(), args[4].as_str());

    // write_complex_vector_r_i(&vector_res, "./Uvych_r.xls","./Uvych_i.xls", 50, 50);
    // write_complex_vector_r_i(&vector1, "./Uvych_orig_r.xls","./Uvych_orig_i.xls", 100, 100);

}

