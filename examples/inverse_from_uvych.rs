use std::path::PathBuf;
use clap::Parser;
use gcg2d::solvers::solve;
use gcg2d::tasks::{only_from_saved_uvych, TaskParameters};

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
    ip1: usize,

    #[clap(long = "ip2")]
    ip2: usize,

}

fn main() {

    let args = Cli::parse();

    let task = only_from_saved_uvych(args.path_from, args.path_save);

    solve(&task, &TaskParameters::base(args.p, args.point, args.ip1, args.ip2));
}