pub mod stream;
pub mod consts;
mod matrix_system;
mod linalg;
mod memory;
mod direct_problem;
mod inverse_problem;
mod initial;
mod neuro;
mod vych;
mod common;

mod tasks;

use gcg2d::solvers::{solve};
use gcg2d::tasks::*;


fn main() -> anyhow::Result<()> {

    // ---------------------------------------------------------------------------------------------------------------



    //let task = init_data_and_full_cycle_with_denoise_k_inv("", "./output", 1e-10);
    //let task = load_k_w_and_full_cycle(dir_1, "./output");
    //let task = only_from_saved_uvych("./input", "./output/trash");

    //let task = only_from_saved_uvych("./output/trash", "./output");
    let settings = init_data_and_full_cycle_with_denoise_J("./output/trash", "./output", 0.0000000095, false);
    //let task = init_data_and_full_cycle("asdf", "./output/trash");

    let mut params = TaskParameters::from_grid(30, 1);
    solve(&settings, &mut params);

    Ok(())

}




