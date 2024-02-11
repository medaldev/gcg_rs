extern crate gcg2d;

use std::path::PathBuf;
use clap::Parser;
use num::integer::Roots;
use gcg2d::common::reduce_matrix;
use gcg2d::stream::{read_complex_vec_from_binary, write_complex_vec_to_binary, write_complex_vector_r_i};


#[derive(Parser)]
struct Cli {
    #[clap(long = "path_from")]
    path_from: PathBuf,

    #[clap(long = "path_save")]
    path_save: PathBuf,

    #[clap(long = "width")]
    width: usize,
    #[clap(long = "height")]
    height: usize,

    #[clap(long = "start")]
    start: usize,

    #[clap(long = "step")]
    step: usize,

}


fn main() -> anyhow::Result<()> {
    let args = Cli::parse();

    let (uvych_r_name, uvych_i_name) = ("Uvych_re", "Uvych_im");
    let vector1 = read_complex_vec_from_binary(args.path_from.join(uvych_r_name), args.path_from.join(uvych_i_name));

    println!("N_in = {:?}", vector1.len());

    let (vector_res, res_shape) = reduce_matrix(vector1.len(), &vector1, args.start, args.step);
    println!("N_out = {:?}", vector_res.len());

    println!("sqrt = {:?}", vector_res.len().sqrt());

    write_complex_vec_to_binary(&vector_res, args.path_save.join(uvych_r_name), args.path_save.join(uvych_i_name));

    println!("Res vector: len={}, shape={:?}", vector_res.len(), res_shape);

    write_complex_vector_r_i(&vector_res,
                             args.path_save.join(format!("{}{}", uvych_r_name, ".xls")),
                             args.path_save.join(format!("{}{}", uvych_i_name, ".xls")),
                             res_shape.1, res_shape.0);



    write_complex_vector_r_i(&vector1,
                             args.path_save.join(format!("{}{}", uvych_r_name, "_orig.xls")),
                             args.path_save.join(format!("{}{}", uvych_i_name, "_orig.xls")),
                             args.width, args.height);

    Ok(())
}

