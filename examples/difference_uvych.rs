use std::path::PathBuf;
use clap::Parser;
use num::complex::{Complex64, ComplexFloat};
use num::Zero;
use gcg2d::memory::create_vector_memory;
use gcg2d::solvers::solve;
use gcg2d::stream::{ComplexVectorSaver, read_complex_vector, write_complex_vector};
use gcg2d::tasks::{only_from_saved_uvych, TaskParameters};

#[derive(Parser)]
struct Cli {
    #[clap(long = "path_first")]
    path_first: PathBuf,

    #[clap(long = "path_second")]
    path_second: PathBuf,

    #[clap(long = "path_save")]
    path_save: PathBuf,

    #[clap(long = "name")]
    name: String,

    #[clap(long = "p")]
    p: usize,

    #[clap(long = "point")]
    point: usize,

    #[clap(long = "ip1")]
    ip1: Option<usize>,

    #[clap(long = "ip2")]
    ip2: Option<usize>,

}

fn main() {

    let args = Cli::parse();


    let mut params = TaskParameters::base(args.p, args.point, args.ip1.unwrap_or(3), args.ip2.unwrap_or(3));


    let mut Uvych1 = create_vector_memory(params.n, Complex64::zero());
    let mut Uvych2 = create_vector_memory(params.n, Complex64::zero());

    read_complex_vector(
        &mut Uvych1,
        args.path_first.join(format!("{}{}{}", args.name.as_str(), ComplexVectorSaver::IM, ".xls")),
        args.path_first.join(format!("{}{}{}", args.name.as_str(), ComplexVectorSaver::IM, ".xls")),
        params.n_x, params.n_y
    );

    read_complex_vector(
        &mut Uvych2,
        args.path_second.join(format!("{}{}{}", args.name.as_str(), ComplexVectorSaver::IM, ".xls")),
        args.path_second.join(format!("{}{}{}", args.name.as_str(), ComplexVectorSaver::IM, ".xls")),
        params.n_x, params.n_y
    );

    let Uvych_diff: Vec<Complex64> = Uvych1.iter().zip(Uvych2.iter()).map(|(u1, u2)| u1 - u2).collect();
    write_complex_vector(
        &Uvych_diff,
        args.path_save.join(format!("{}_diff{}{}", args.name.as_str(), ComplexVectorSaver::ABS, ".xls")),
        params.n_x, params.n_y
    );



}