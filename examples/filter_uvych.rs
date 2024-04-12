use std::fs;
use std::path::Path;
use num::complex::Complex64;
use num::Zero;
use gcg2d::common::{add_noise_to_matrix, build_complex_vector, get_noised_tensor, matrix_to_vec, separate_re_im, vec_to_matrix};
use gcg2d::memory::create_vector_memory;
use gcg2d::neuro;
use gcg2d::neuro::xls_to_matrix;
use gcg2d::stream::ComplexVectorSaver;
use gcg2d::stream::SaveFormat::{Bin, Csv, Xls};
use gcg2d::tasks::TaskParameters;

fn main() {

    let params = TaskParameters::from_grid(80, 1);

    let vector_stream = ComplexVectorSaver::init(
        Path::new("/home/amedvedev/projects/python/DenoisingCNN/data/datasets/gcg19/train/calculations/0a29a46e-cedb-4b67-8423-59bbb0b7a2bd/"),
        Path::new("./")
    );


    let mut Uvych = create_vector_memory(params.n, Complex64::zero());

    vector_stream.load(&mut Uvych, params.n, "Uvych2", Bin, &params);


    let parts_uvych = separate_re_im(&Uvych);


    let n_noised = 64;
    let noised_uvych_re_tensor = get_noised_tensor(&vec_to_matrix(&parts_uvych.0, params.n_x, params.n_y), n_noised, 1.0e-18);
    let noised_uvych_im_tensor = get_noised_tensor(&vec_to_matrix(&parts_uvych.1, params.n_x, params.n_y), n_noised, 1.0e-18);


    let model_denoiser = Path::new("./models/uvych_tensor_denoiser_2_cpu_traced.pt");

    save_noised_tensor(&params, &vector_stream, "Uvych2_re", "xls", 64, 0.1);
    let example_tensor_vec: Vec<Vec<Vec<f64>>> = serde_json::from_str(fs::read_to_string(vector_stream.output_dir.join("Uvych2_re_noised.tensor")).unwrap().as_str()).unwrap();
    //let example_tensor_vec: Vec<Vec<Vec<f64>>> = serde_json::from_str(fs::read_to_string(vector_stream.input_dir.join("Uvych2_abs_noised.tensor")).unwrap().as_str()).unwrap();



    let cleaned_re = matrix_to_vec(
        neuro::denoise_tensor(example_tensor_vec, (n_noised, params.n_y, params.n_x), model_denoiser).unwrap(),
        params.n_x, params.n_y
    );
    let cleaned_im = matrix_to_vec(
        neuro::denoise_tensor(noised_uvych_im_tensor, (n_noised, params.n_y, params.n_x), model_denoiser).unwrap(),
        //noised_uvych_im_tensor[0].clone(),
        params.n_x, params.n_y);
    let Uvych_filtered = build_complex_vector(params.n, cleaned_re, cleaned_im);


    // Запись зашумлённых данных
    vector_stream.save(&Uvych_filtered, "Uvych", &[Xls], &params);
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