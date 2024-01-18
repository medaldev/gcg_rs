extern crate gcg2d;

use std::path::PathBuf;
use std::time::Instant;
use clap::Parser;
use num::integer::Roots;
use gcg2d::common::reduce_matrix;
use gcg2d::stream::{read_complex_vec_from_binary, write_complex_vec_to_binary, write_complex_vector_r_i};


#[derive(Parser)]
struct Cli {
    #[clap(long = "path_data")]
    path_data: PathBuf,

    #[clap(long = "path_save")]
    path_save: PathBuf,

    #[clap(long = "width")]
    width: usize,
    #[clap(long = "height")]
    height: usize,

    #[clap(long = "model")]
    model: PathBuf,



}


fn main() -> anyhow::Result<()> {
    let args = Cli::parse();

    let (uvych_r_name, uvych_i_name) = ("Uvych_r", "Uvych_i");
    let vector1 = read_complex_vec_from_binary(args.path_data.join(uvych_r_name), args.path_data.join(uvych_i_name));

    // Подавление шума с помощью нейронной сети
    let start_time = Instant::now();

    println!("Using neural network model to denoise K_inv.");

    let denoised = neuro::run_im(&data, params.n, params.n_x, params.n_y, model_path, true).unwrap();
    write_complex_vector(&denoised, settings.output_dir.join(format!("{}_denoised.xls", data_label)), params.n_x, params.n_y);

    println!(">> Model inference time: {:?} seconds", start_time.elapsed().as_secs());

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

