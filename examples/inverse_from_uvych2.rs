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

}

fn main() {

    let args = Cli::parse();

    let task = only_from_saved_uvych(args.path_from.as_path(), args.path_save.as_path());

    solve(&task, &mut TaskParameters::load_from_file(args.path_from.join("params.json").as_path()).unwrap());
}