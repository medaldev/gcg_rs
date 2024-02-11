use std::fmt::format;
use std::fs;
use std::path::{Path, PathBuf};
use itertools::izip;
use num::complex::Complex64;
use num::Zero;
use pbr::ProgressBar;
use gcg2d::common::{add_noise, add_noise_re_im, add_noise_to_matrix, build_complex_vector, matrix_to_vec, separate_re_im};
use gcg2d::{initial, vych};
use gcg2d::memory::{create_matrix_memory, create_vector_memory};
use gcg2d::neuro::xls_to_matrix;
use gcg2d::solvers::{solve};
use gcg2d::stream::{ComplexVectorSaver, matrix_to_file, read_xls, write_f64_to_file};
use gcg2d::stream::SaveFormat::{Bin, Csv, Xls};
use gcg2d::tasks::{load_k_w_and_full_cycle, SolutionSettings, TaskParameters};

fn main() -> anyhow::Result<()> {

    // ---------------------------------------------------------------------------------------------------------------

    let mut settings = load_k_w_and_full_cycle("<None>", "<None>");


    for type_data in ["train", "val"] {

        let data_dir = PathBuf::from("/home/amedvedev/projects/python/DenoisingCNN/data/datasets/gcg19").join(type_data);
        let calc_dir = data_dir.join("calculations");

        // let clear_dir = data_dir.join("clear");
        // let noised_dir = data_dir.join("noised");

        let files = match fs::read_dir(calc_dir.as_path()) {
            Ok(real_files) => {real_files}
            Err(_) => {
                return Err(anyhow::anyhow!("Provided path is not correct {:?}", calc_dir.as_path()))
            }
        };

        let n = fs::read_dir(calc_dir.as_path()).unwrap().count();

        let mut pb = ProgressBar::new(n as u64);
        pb.set_width(Some(75));

        for name_res in files {

            pb.inc();

            let name = name_res.unwrap().file_name().to_str().unwrap().to_string();
            let task_dir = calc_dir.join(name.as_str());

            // fs::create_dir_all(clear_dir.as_path())?;
            // fs::create_dir_all(noised_dir.as_path())?;

            settings.input_dir = task_dir.clone();
            settings.output_dir = task_dir.clone();

            let vector_stream = ComplexVectorSaver::init(settings.input_dir.as_path(), settings.output_dir.as_path());

            let params = TaskParameters::load_from_file(settings.input_dir.join("params.json").as_path())?;


            //save_left_right_parts_of(task_dir.as_path(), &params, "Uvych_abs", "xls")?;

            //save_bvych(&params, &vector_stream);
            //save_left_right_parts_of(task_dir.as_path(), &params, "Bvych_abs", "xls")?;
            //save_left_right_parts_of(task_dir.as_path(), &params, "Uvych_noised_abs", "xls")?;

            //save_noised_uvych(&params, &vector_stream);

            save_noised_vector(&params, &vector_stream, "K", 1e-4);

            // fs::copy(task_dir.join("Uvych_re.xls").as_path(), clear_dir.join(format!("{}.xls", name)).as_path()).unwrap();
            // fs::copy(task_dir.join("Uvych_abs.xls").as_path(), noised_dir.join(format!("{}.xls", name)).as_path()).unwrap();


        }

        pb.finish_println("");

    }

    Ok(())
}

fn save_left_right_parts_of(task_dir: &Path, params: &TaskParameters, namefile: &str, ext: &str) -> anyhow::Result<()> {
    let Uvych_abs = xls_to_matrix(task_dir.join(format!("{}.{}", namefile, ext)));

    let mut Uvych_left = vec![vec![0.0; params.n_x / 2]; params.n_y];
    let mut Uvych_right = vec![vec![0.0; params.n_x / 2]; params.n_y];

    for i in 0..params.n_y {
        for j in 0..params.n_x / 2 {
            Uvych_left[i][j] = Uvych_abs[i][j];
            Uvych_right[i][j] = Uvych_abs[i][params.n_x / 2 + j];
        }

    }

    matrix_to_file(&Uvych_left, task_dir.join(format!("{}_left.{}", namefile, ext)).as_path())?;
    matrix_to_file(&Uvych_right, task_dir.join(format!("{}_right.{}", namefile, ext)).as_path())?;


    Ok(())
}

fn save_noised_uvych(params: &TaskParameters, vector_stream: &ComplexVectorSaver) {

    let mut Uvych = create_vector_memory(params.n, Complex64::zero());

    vector_stream.load(&mut Uvych, params.n, "Uvych", Bin, &params);

    let mut Uvych_noised = Uvych.clone();
    add_noise(&mut Uvych_noised, 1e-4);

    // let mut Uvych_div = create_vector_memory(params.n, Complex64::zero());
    //
    // for i in 0..params.n {
    //     Uvych_div[i] = Complex64::new((Uvych_noised[i].re - Uvych[i].re),
    //                                   (Uvych_noised[i].im - Uvych[i].im) / Uvych[i].im)
    // }

    //Запись зашумлённых данных
    vector_stream.save(&Uvych_noised, "Uvych_noised", &[Xls], &params);
    // vector_stream.save(&Uvych_div, "Uvych_div", &[Xls], &params);
}

fn save_noised_vector(params: &TaskParameters, vector_stream: &ComplexVectorSaver, namefile: &str, pct: f64) {

    let mut vector = create_vector_memory(params.n, Complex64::zero());

    vector_stream.load(&mut vector, params.n, namefile, Xls, &params);

    let mut vector_noised = vector.clone();
    add_noise(&mut vector_noised, pct);

    vector_stream.save(&vector_noised, format!("{}_noised", namefile).as_str(), &[Xls], &params);
}

fn save_bvych(params: &TaskParameters, vector_stream: &ComplexVectorSaver) {
    let mut Bvych = create_vector_memory(params.n, Complex64::zero());
    vych::r_part_vych(params.point, params.shift, params.n_x, params.n_y, params.dim_x, params.dim_y, params.a, params.b, params.k0, params.n, params.ip1, &mut Bvych);
    vector_stream.save(&Bvych, "Bvych", &[Xls], &params);


}

