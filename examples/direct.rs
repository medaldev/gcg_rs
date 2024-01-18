use std::path::PathBuf;
use clap::{arg, Parser};
use gcg2d::solvers::solve;
use gcg2d::tasks::{init_data_and_full_cycle, TaskParameters};

#[derive(Parser)]
struct Cli {
    #[clap(long = "path_from")]
    path_from: PathBuf,

    #[clap(long = "path_save")]
    path_save: PathBuf,

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

    let settings = init_data_and_full_cycle(args.path_from, args.path_save);

    solve(&settings, &TaskParameters::base(args.p, args.point, args.ip1.unwrap_or(3), args.ip2.unwrap_or(3)));
}