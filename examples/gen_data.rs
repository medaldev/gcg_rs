use std::fs;
use std::path::{Path, PathBuf};
use itertools::izip;
use gcg2d::initial;
use gcg2d::solvers::SolutionSettings;
use gcg2d::tasks::TaskParameters;

fn main() {

    // ---------------------------------------------------------------------------------------------------------------

    let mut settings = SolutionSettings {
        // Задание начальных значений K
        use_initial_k: false,

        // Загрузка K из файлов
        load_init_k_w_from_files: true,

        // Решить прямую задачу
        solve_direct: true,

        // Загрузка J из файлов
        load_j_from_files: false,

        // Внесение шума в J
        add_noise_j: false,
        pct_noise_j: 0.0,

        // Использовать нейросеть для очистки J
        neuro_use_j: false,
        neuro_use_k_inv: false,

        // Расчёт поля в точках наблюдения
        vych_calc: true,

        // Загрузка Uvych из файлов
        load_uvych_from_files: false,

        // Внесение шума в Uvych
        add_noise_uvych: false,
        pct_noise_uvych: 0.0,

        // Решить обратную задачу
        solve_inverse: true,

        // Пути к файлам
        input_dir: PathBuf::from("./input"),
        output_dir: PathBuf::from("./output"),
    };

    for type_data in ["train", "val"] {
        let data_dir = PathBuf::from("/home/amedvedev/fprojects/python/denoising/data/datasets/gcg18").join(type_data).join("clear");
        let res_dir = PathBuf::from("/home/amedvedev/fprojects/python/denoising/data/datasets/gcg18").join(type_data).join("noised");

        println!("{:?}", data_dir);
        println!("{:?}", res_dir);
        let k_r_dir = data_dir.join("txt");
        let k_i_dir = data_dir.join("txt_im");

        let w_r_dir = data_dir.join("txtW_re");
        let w_i_dir = data_dir.join("txtW_im");

        let in_dir = data_dir.join("input");
        let out_dir = data_dir.join("calculations");

        let files_set = izip!(
            fs::read_dir(k_r_dir.as_path()).unwrap(),
            fs::read_dir(k_i_dir.as_path()).unwrap(),
            fs::read_dir(w_r_dir.as_path()).unwrap(),
            fs::read_dir(w_i_dir.as_path()).unwrap()
        );


        for files_objs in files_set {
            let file_k_r = files_objs.0.unwrap();
            let file_k_i = files_objs.1.unwrap();
            let file_w_r = files_objs.2.unwrap();
            let file_w_i = files_objs.3.unwrap();

            let file_name = PathBuf::from(file_k_r.file_name()).with_extension("");

            let task_in_dir = in_dir.join(file_name.as_path());
            let task_out_dir = out_dir.join(file_name.as_path());

            // copy_input_data(file_k_r.path().as_path(), task_in_dir.as_path(), task_out_dir.as_path(), "K_r");
            // copy_input_data(file_k_i.path().as_path(), task_in_dir.as_path(), task_out_dir.as_path(), "K_i");
            //
            // copy_input_data(file_w_r.path().as_path(), task_in_dir.as_path(), task_out_dir.as_path(), "W_r");
            // copy_input_data(file_w_i.path().as_path(), task_in_dir.as_path(), task_out_dir.as_path(), "W_i");


            settings.input_dir = task_in_dir.clone();
            settings.output_dir = task_out_dir.clone();

            let p = 30;
            let point = 2;
            let params = TaskParameters::from_grid(p, point);

            let k0 = 0.1;

            let mut surface = initial::Surface::new(p, p, point, k0);
            let total_figs = initial::polygon_covering(
                &mut surface, 0.025, 0.4, 100,
                (0.99, 1.01), 0.4, 0.4, 50
            ).unwrap();



            // solve(&task);

            // Uvych_r.xls

            // set feature (noised)
            fs::copy(task_out_dir.join("Uvych_r.xls"), res_dir.join("txt").join(file_k_r.file_name())).unwrap();

            //set target (clear)
            fs::copy(task_out_dir.join("W.xls"), data_dir.join("txt").join(file_k_r.file_name())).unwrap();

            println!("{:?}", file_k_r.path().file_name().unwrap());
        }

        //fs::remove_dir_all(temp_input_dir.as_path()).unwrap();

        //solve(task);
    }
}


