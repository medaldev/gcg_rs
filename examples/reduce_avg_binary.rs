extern crate gcg2d;

use std::path::PathBuf;
use num::integer::Roots;
use gcg2d::common::{avg_pool, reduce_matrix};
use gcg2d::stream::{ComplexVectorSaver, read_complex_vec_from_binary, write_complex_vec_to_binary, write_complex_vector_r_i};

use clap::Parser;


#[derive(Parser)]
struct Cli {

    #[clap(long="path_from")]
    path_from: PathBuf,

    #[clap(long="path_save")]
    path_save: PathBuf,

    #[clap(long="width")]
    width: usize,
    #[clap(long="height")]
    height: usize,

    #[clap(long="window_width")]
    window_width: usize,
    #[clap(long="window_height")]
    window_height: usize,

    #[clap(long="stride_y")]
    stride_y: Option<usize>,
    #[clap(long="stride_x")]
    stride_x: Option<usize>,

}


fn main() -> anyhow::Result<()> {

    let args = Cli::parse();

    let (uvych_r_name, uvych_i_name) = ("Uvych_re", "Uvych_im");
    let vector1 = read_complex_vec_from_binary(args.path_from.join(uvych_r_name), args.path_from.join(uvych_i_name));


    println!("N_in = {:?}", vector1.len());

    let stride = (
        match args.stride_y {
            None => {1}
            Some(val) => {val}
        },
        match args.stride_x {
            None => {1}
            Some(val) => {val}
        }
    );

    let (vector_res, res_shape) = avg_pool(&vector1, (args.height, args.width), (args.window_height, args.window_width), stride);
    println!("N_out = {:?}", vector_res.len());

    println!("sqrt = {:?}", vector_res.len().sqrt());

    write_complex_vec_to_binary(&vector_res, args.path_save.join(uvych_r_name), args.path_save.join(uvych_i_name));

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


