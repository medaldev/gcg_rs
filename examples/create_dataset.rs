use std::fmt::format;
use std::fs;
use std::path::{Path, PathBuf};
use itertools::izip;
use num::complex::Complex64;
use num::Zero;
use gcg2d::common::{add_noise, add_noise_re_im, build_complex_vector, matrix_to_vec, separate_re_im};
use gcg2d::initial;
use gcg2d::memory::create_vector_memory;
use gcg2d::solvers::{solve};
use gcg2d::stream::{ComplexVectorSaver, write_f64_to_file};
use gcg2d::stream::SaveFormat::{Bin, Csv, Xls};
use gcg2d::tasks::{load_k_w_and_full_cycle, SolutionSettings, TaskParameters};

fn main() -> anyhow::Result<()> {

    // ---------------------------------------------------------------------------------------------------------------

    let mut settings = load_k_w_and_full_cycle("<None>", "<None>");


    for type_data in ["train", "val"] {

        let data_dir = PathBuf::from("/home/amedvedev/fprojects/python/denoising/data/datasets/gcg19").join(type_data);
        let calc_dir = data_dir.join("calculations");

        let clear_dir = data_dir.join("clear");
        let noised_dir = data_dir.join("noised");

        let files = match fs::read_dir(calc_dir.as_path()) {
            Ok(real_files) => {real_files}
            Err(_) => {
                return Err(anyhow::anyhow!("Provided path is not correct {:?}", calc_dir.as_path()))
            }
        };

        for name_res in files {

            let name = name_res.unwrap().file_name().to_str().unwrap().to_string();
            let task_dir = calc_dir.join(name.as_str());

            fs::create_dir_all(clear_dir.as_path())?;
            fs::create_dir_all(noised_dir.as_path())?;

            settings.input_dir = task_dir.clone();
            settings.output_dir = task_dir.clone();

            let vector_stream = ComplexVectorSaver::init(settings.input_dir.as_path(), settings.output_dir.as_path());

            let mut params = TaskParameters::load_from_file(settings.input_dir.join("params.json").as_path())?;

            fs::copy(task_dir.join("Uvych_re.xls").as_path(), clear_dir.join(format!("{}.xls", name)).as_path()).unwrap();

            let mut Uvych = create_vector_memory(params.n, Complex64::zero());

            vector_stream.load(&mut Uvych, params.n, "Uvych", Bin, &params);

            add_noise(&mut Uvych, 1e-4);

            // Запись зашумлённых данных
            vector_stream.save(&Uvych, "Uvych_noised", &[Xls, Csv], &params);

            fs::copy(task_dir.join("Uvych_noised_re.xls").as_path(), noised_dir.join(format!("{}.xls", name)).as_path()).unwrap();
            println!("asdf");


        }

    }

    Ok(())
}

