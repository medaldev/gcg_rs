use std::path::PathBuf;
use clap::Parser;
use gcg2d::solvers::solve;
use gcg2d::tasks::{init_data_and_forward, init_data_and_full_cycle, TaskParameters};


#[derive(Parser)]
struct Cli {

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

    let mut params = TaskParameters::base(args.p, args.point, args.ip1.unwrap_or(3), args.ip2.unwrap_or(3));

    let res = solve(&init_data_and_full_cycle("./<>", args.path_save.to_str().unwrap()), &mut params);
}