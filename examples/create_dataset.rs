use std::fmt::format;
use std::fs;
use std::path::{Path, PathBuf};
use itertools::izip;
use num::complex::{Complex64, ComplexFloat};
use num::Zero;
use pbr::ProgressBar;
use rand::distributions::{Distribution, Uniform};
use gcg2d::common::{add_noise, add_noise_re_im, add_noise_to_matrix, build_complex_vector, matrix_to_vec, rotate_matrix, separate_re_im};
use gcg2d::{initial, vych};
use gcg2d::linalg::min_max_f64_vec;
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

        let mut global_max: Option<f64> = None;
        let mut global_min: Option<f64> = None;

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

            // save_noised_vector(&params, &vector_stream, "Uvych2", 0.01, false);
            save_noised_vector(&params, &vector_stream, "J", 0.30, true);
            //save_rotated_matrix_pair_proba(&params, &vector_stream, "Uvych2_re","Uvych2_noised_re", "xls", 0.5);


            //calc_uvych_bvych(&params, &vector_stream);

            //save_noised_tensor(&params, &vector_stream, "K_abs", "xls", 64, 0.1);
            //save_noised_tensor(&params, &vector_stream, "Uvych2_re", "xls", 32, 0.001);
            // save_rotated_matrix_pair_proba(&params, &vector_stream, "Uvych2_re",
            //                                vec!["Uvych2_noised_re", "Uvych2_noise_only_re"], "xls", 0.5);


            let some_vec = xls_to_matrix(task_dir.join(format!("{}.{}", "J_re", "xls")));

            // let k_sum = some_vec.concat().iter().sum::<f64>() - params.k0.re * params.n as f64;
            //
            // if k_sum.abs() < 0.000001 {
            //     println!("{:?}, {}", task_dir, k_sum);
            //     //fs::remove_dir_all(task_dir).unwrap();
            // }

            let (max_uv, min_uv) = min_max_f64_vec(&some_vec.concat());


            global_max = match global_max {
                None => {Some(max_uv)},
                Some(el) => {
                    if max_uv > el {Some(max_uv)} else { Some(el) }
                }
            };

            global_min = match global_min {
                None => {Some(min_uv)},
                Some(el) => {
                    if min_uv < el {Some(min_uv)} else { Some(el) }
                }
            }

            // fs::copy(task_dir.join("Uvych_re.xls").as_path(), clear_dir.join(format!("{}.xls", name)).as_path()).unwrap();
            // fs::copy(task_dir.join("Uvych_abs.xls").as_path(), noised_dir.join(format!("{}.xls", name)).as_path()).unwrap();


        }

        pb.finish_println("");

        println!("Min: {:?}", global_min);
        println!("Max: {:?}", global_max);

    }

    Ok(())
}


fn save_rotated_matrix_proba(params: &TaskParameters, vector_stream: &ComplexVectorSaver, namefile: &str, ext: &str, probability: f64)  {
    let mut some_vec = xls_to_matrix(vector_stream.input_dir.join(format!("{}.{}", namefile, ext)));
    let mut rng = rand::thread_rng();
    let ppp = Uniform::from(0.0..1.0);
    if ppp.sample(&mut rng) <= probability {
        some_vec = rotate_matrix(some_vec);
    }
    matrix_to_file(&some_vec, vector_stream.output_dir.join(format!("{}_rotated.{}", namefile, ext)).as_path()).unwrap();

}

fn save_rotated_matrix_pair_proba(params: &TaskParameters, vector_stream: &ComplexVectorSaver, name1: &str, names: Vec<&str>, ext: &str, probability: f64)  {
    let mut feature_vec = xls_to_matrix(vector_stream.input_dir.join(format!("{}.{}", name1, ext)));
    let mut rng = rand::thread_rng();
    let ppp = Uniform::from(0.0..1.0);
    let rotate_condition =ppp.sample(&mut rng) <= probability;
    if rotate_condition {
        feature_vec = rotate_matrix(feature_vec);
    }
    matrix_to_file(&feature_vec, vector_stream.output_dir.join(format!("{}_rotated.{}", name1, ext)).as_path()).unwrap();

    for name2 in names {
        let mut target_vec = xls_to_matrix(vector_stream.input_dir.join(format!("{}.{}", name2, ext)));
        if rotate_condition {
            target_vec = rotate_matrix(target_vec);
        }
        matrix_to_file(&target_vec, vector_stream.output_dir.join(format!("{}_rotated.{}", name2, ext)).as_path()).unwrap();

    }

}

fn save_noised_tensor(params: &TaskParameters, vector_stream: &ComplexVectorSaver, namefile: &str, ext: &str, n_noised: usize, pct: f64)  {
    let some_vec = xls_to_matrix(vector_stream.input_dir.join(format!("{}.{}", namefile, ext)));
    let mut noised_vec = Vec::with_capacity(n_noised);
    for i in 0..n_noised {
        let mut vec_clone = some_vec.clone();
        add_noise_to_matrix(&mut vec_clone, pct);
        noised_vec.push(vec_clone);
    }

    let tensor_json = serde_json::to_string(&noised_vec).unwrap();
    fs::write(vector_stream.output_dir.join(format!("{}_noised.{}", namefile, "tensor")), tensor_json).expect("Unable to write file");

}
fn save_noised_k_no_k0(params: &TaskParameters, vector_stream: &ComplexVectorSaver, namefile: &str)  {
    let mut K_no_k0_noised = load_vector(&params, &vector_stream, "K");
    for num in K_no_k0_noised.iter_mut() {
        *num -= params.k0;
    }
    vector_stream.save(&K_no_k0_noised, "K-k0", &[Xls], &params);

    if K_no_k0_noised.iter().sum::<Complex64>().abs() > 0.0 {
        add_noise_re_im(&mut K_no_k0_noised, 0.8);
    }
    vector_stream.save(&K_no_k0_noised, "K-k0_noised", &[Xls], &params);
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

fn save_noised_vector(params: &TaskParameters, vector_stream: &ComplexVectorSaver, namefile: &str, pct: f64, noise_both: bool) {

    let mut vector = create_vector_memory(params.n, Complex64::zero());

    vector_stream.load(&mut vector, params.n, namefile, Xls, &params);

    let mut only_noise = create_vector_memory(params.n, Complex64::zero());

    let mut vector_noised = vector.clone();
    if noise_both{
        add_noise_re_im(&mut vector_noised, pct);
    }
    else {
        add_noise(&mut vector_noised, pct);
    }

    for i in 0..params.n {
        only_noise[i] = vector_noised[i] - vector[i];
    }

    vector_stream.save(&vector_noised, format!("{}_noised", namefile).as_str(), &[Xls], &params);
    vector_stream.save(&only_noise, format!("{}_noise_only", namefile).as_str(), &[Xls], &params);
}

fn load_vector(params: &TaskParameters, vector_stream: &ComplexVectorSaver, namefile: &str) -> Vec<Complex64>{
    let mut vector = create_vector_memory(params.n, Complex64::zero());
    vector_stream.load(&mut vector, params.n, namefile, Xls, &params);
    vector
}

fn save_bvych(params: &TaskParameters, vector_stream: &ComplexVectorSaver) {
    let mut Bvych = create_vector_memory(params.n, Complex64::zero());
    vych::r_part_vych(params.point, params.shift, params.n_x, params.n_y, params.dim_x, params.dim_y, params.a, params.b, params.k0, params.n, params.ip1, &mut Bvych);
    vector_stream.save(&Bvych, "Bvych", &[Xls], &params);


}

fn calc_uvych_bvych(params: &TaskParameters, vector_stream: &ComplexVectorSaver) {
    let mut Bvych = create_vector_memory(params.n, Complex64::zero());
    let mut Uvych = create_vector_memory(params.n, Complex64::zero());
    vych::r_part_vych(params.point, params.shift, params.n_x, params.n_y, params.dim_x, params.dim_y, params.a, params.b, params.k0, params.n, params.ip1, &mut Bvych);
    vector_stream.save(&Bvych, "Bvych2", &[Xls], &params);

    let mut J = create_vector_memory(params.n, Complex64::new(0.1, 0.0));
    vector_stream.load(&mut J, params.n, "J", Xls, params);

    vych::get_uvych(params.point, params.n, params.n_x, params.n_y, params.ip1, params.dim_x, params.dim_y, params.a, params.b, params.shift,
                    &J, &mut Uvych, &Bvych, params.k0);

    vector_stream.save(&Uvych, "Uvych2", &[Xls, Bin], params);

}



