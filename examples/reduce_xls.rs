use num::complex::Complex64;
extern crate gcg2d;

use std::env;
use num::integer::Roots;
use gcg2d::common::reduce_matrix;
use gcg2d::stream::{get_complex_vector, write_complex_vector_r_i};

fn main() {
    let args: Vec<String> = env::args().collect();

    assert_eq!(args.len(), 6);
    println!("{:?}", args);

    let vector1 = get_complex_vector(args[1].as_str(), args[2].as_str());

    println!("N_in = {:?}", vector1.len());

    let vector_res = reduce_matrix(vector1.len(), &vector1, str::parse::<usize>(args[5].as_str()).unwrap());
    println!("N_out = {:?}", vector_res.len());


    let size = vector_res.len().sqrt();
    println!("sqrt = {:?}", size);

    write_complex_vector_r_i(&vector_res, args[3].as_str(), args[4].as_str(), size, size);

}

