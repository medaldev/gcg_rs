use std::fs;
use std::path::{Path, PathBuf};
use clap::Parser;
use itertools::izip;
use num::complex::Complex64;
use rand::distributions::Distribution;
use gcg2d::common::{build_complex_vector, matrix_to_vec};
use gcg2d::initial;
use gcg2d::solvers::{solve};
use gcg2d::stream::{ComplexVectorSaver, write_f64_to_file};
use gcg2d::stream::SaveFormat::{Csv, Xls};
use gcg2d::tasks::{load_k_w_and_full_cycle, SolutionSettings, TaskParameters};

#[derive(Parser)]
struct Cli {
    #[clap(long = "data_dir")]
    data_dir: PathBuf,

    #[clap(long = "from")]
    from: usize,

    #[clap(long = "to")]
    to: usize,

}

fn main() {

    let args = Cli::parse();

    // ---------------------------------------------------------------------------------------------------------------

    let mut settings = load_k_w_and_full_cycle("<None>", "<None>");
    settings.solve_inverse = false;

    let mut rng = rand::thread_rng();
    let gen_k0 = rand::distributions::Uniform::from(0.1..50.1);
    let gen_poly_size = rand::distributions::Uniform::from(0.02..0.1);
    let gen_k0_dev = rand::distributions::Uniform::from(0.01..0.3);
    let gen_irr = rand::distributions::Uniform::from(0.1..0.99);
    let gen_proba = rand::distributions::Uniform::from(0.2..0.99);
    let gen_spikiness = rand::distributions::Uniform::from(0.1..0.99);
    let gen_vert = rand::distributions::Uniform::from(10usize..70);

    for itera in args.from..args.to {

        for type_data in ["train", "val"] {

            let data_dir = args.data_dir.join(type_data).join("calculations");

            let name = format!("example_{itera}");

            let task_dir = data_dir.join(name);

            fs::create_dir_all(task_dir.as_path()).unwrap();


            settings.input_dir = task_dir.clone();
            settings.output_dir = task_dir.clone();

            let vector_stream = ComplexVectorSaver::init(settings.input_dir.as_path(), settings.output_dir.as_path());


            let p = 40;
            let point = 2;
            let k0 = gen_k0.sample(&mut rng);

            let mut params = TaskParameters::from_grid(p, point)
                .set_k0(Complex64::new(k0, 0.0));


            let mut surface = initial::Surface::new(p, p, point, k0);
            let total_figs = initial::polygon_covering(
                &mut surface,
                gen_poly_size.sample(&mut rng),
                gen_proba.sample(&mut rng),
                1000,
                gen_k0_dev.sample(&mut rng),
                gen_irr.sample(&mut rng),
                gen_spikiness.sample(&mut rng),
                gen_vert.sample(&mut rng)
            ).unwrap();

            let K = build_complex_vector(
                params.n,
                matrix_to_vec(surface.get_k_matrix(), surface.cols, surface.rows),
                vec![0.0; surface.cols * surface.rows],
            );

            let W = build_complex_vector(
                params.n,
                matrix_to_vec(surface.get_w_matrix(), surface.cols, surface.rows),
                vec![0.0; surface.cols * surface.rows],
            );

            write_f64_to_file(params.k0.re, settings.output_dir.join("k0_re.txt").as_path()).unwrap();
            write_f64_to_file(params.k0.im, settings.output_dir.join("k0_im.txt").as_path()).unwrap();
            vector_stream.save(&K, "K", &[Xls, Csv], &params);
            vector_stream.save(&W, "W", &[Xls, Csv], &params);


            solve(&settings, &mut params);



        }

    }
}

