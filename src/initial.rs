use num::complex::Complex64;
use num::{Complex, Zero};
use crate::matrix_system::fill_xy_col;
use crate::memory::create_vector_memory;

use std::f64::consts::PI;
use std::path::Path;
use anyhow::anyhow;
use rand::distributions::{Distribution, Uniform};
use rand_distr::{Normal};

pub fn initial_k0(n: usize, n_x: usize, n_y: usize, dim_x: f64, dim_y: f64, a: f64, b: f64, k1: Complex64,
                  K: &mut [Complex64], W: &mut [Complex64]) {

    let (mut x, mut y) = (0.0, 0.0);

    let mut xc = create_vector_memory(n, 0.0f64);
    let mut yc = create_vector_memory(n, 0.0f64);

    fill_xy_col(n, n_x, n_y, dim_x, dim_y, a, b, &mut xc, &mut yc);

    let r1 = dim_x / 2.0;
    let r2 = 3.* dim_x / 8.0;
    let r3 = dim_x / 4.0;
    let r4 = dim_x / 8.0;
    let mut ind = 0;

    for i in 0..n_x {
        for j in 0..n_y {

            x = xc[ind];
            y = yc[ind];

            K[ind] = Complex64::new(0.7, 0.);
            if x < 0.0 {
                K[ind] = Complex64::new(0.6, 0.);
            }

            W[ind] = Complex64::new(1.0, 0.);

            ind += 1;
        }
    }

    ind = 0;
    for i in 0..n_x {
        for j in 0..n_y {

            x = xc[ind];
            y = yc[ind];

            if x * x / 4.0 + y * y <= r1 * r1 {
                K[ind] = Complex64::new(0.7, 0.);
                if (x < 0.){
                    K[ind] = Complex64::new(0.6, 0.);
                }

                W[ind] = Complex64::new(1.0, 0.);

                if (x * x / 4.0 + y * y <= r2 * r2) {
                    K[ind] = k1;
                    W[ind] = Complex64::zero();

                    if (x * x / 4.0 + y * y <= 1.0) {
                        K[ind] = Complex64::new(0.35, 0.);
                        if (x < 0.){
                            K[ind] = Complex64::new(0.25, 0.);
                        }

                        W[ind] = Complex64::new(1.0, 0.);
                        if (x * x / 4.0 + y * y <= r4 * r4) {
                            K[ind] = k1;
                            W[ind] = Complex64::zero();
                        }

                    }
                }
            }
            else {
                K[ind] = k1;
                W[ind] = Complex64::zero();
            }

            ind += 1;
        }
    }


    for i1 in 0..n {
        K[i1] *= W[i1];
    }

}


pub fn initial_k_polygons(params: &TaskParameters) -> (Vec<Complex<f64>>, Vec<Complex<f64>>) {

    let mut rng = rand::thread_rng();
    let gen_k0 = rand::distributions::Uniform::from(0.1..50.1);
    let gen_poly_size = rand::distributions::Uniform::from(0.02..0.1);
    let gen_k0_dev = rand::distributions::Uniform::from(0.01..0.3);
    let gen_irr = rand::distributions::Uniform::from(0.1..0.99);
    let gen_proba = rand::distributions::Uniform::from(0.2..0.99);
    let gen_spikiness = rand::distributions::Uniform::from(0.1..0.99);
    let gen_vert = rand::distributions::Uniform::from(10usize..70);

    let mut surface = Surface::new(params.p, params.p, params.point, params.k0.re);
    let mut total_figs = vec![];

    while total_figs.is_empty() {
        total_figs = initial::polygon_covering(
            &mut surface,
            gen_poly_size.sample(&mut rng),
            gen_proba.sample(&mut rng),
            1000,
            gen_k0_dev.sample(&mut rng),
            gen_irr.sample(&mut rng),
            gen_spikiness.sample(&mut rng),
            gen_vert.sample(&mut rng)
        ).unwrap();
    }

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
    (K, W)


}

pub fn generate_polygon
(
    center: (f64, f64),
    avg_radius: f64,
    irregularity: f64,
    spikiness: f64,
    num_vertices: usize
)
    -> anyhow::Result<Vec<(f64, f64)>>
{

    if irregularity < 0. || irregularity > 1. {
        return Err(anyhow!("Irregularity must be between 0 and 1."))
    }

    if spikiness < 0. || spikiness > 1. {
        return Err(anyhow!("Spikiness must be between 0 and 1."))
    }

    let irregularity = irregularity * 2. * PI / num_vertices as f64;
    let spikiness = spikiness * avg_radius;
    let angle_steps = random_angle_steps(num_vertices, irregularity);


    let mut points = vec![];
    let mut rng = rand::thread_rng();
    let angle_range = Uniform::from(0.0..2. * PI);
    let mut angle = angle_range.sample(&mut rng);

    let rad_normal = Normal::new(avg_radius, spikiness).unwrap();

    for i in 0..num_vertices {
        let radius = clip(rad_normal.sample(&mut rng), 0., 2. * avg_radius);
        let point = (center.0 + radius * angle.cos(),
                     center.1 + radius * angle.sin());
        points.push(point);
        angle += angle_steps[i];
    }

    Ok(points)
}

pub fn random_angle_steps(steps: usize, irregularity: f64) -> Vec<f64> {
    let mut angles = vec![];
    let lower = (2. * PI / steps as f64) - irregularity;
    let upper = (2. * PI / steps as f64) + irregularity;
    let mut cumsum = 0.0;

    let mut rng = rand::thread_rng();
    let angle_range = Uniform::from(lower..upper);

    for _i in 0..steps {
        let angle = angle_range.sample(&mut rng);
        angles.push(angle);
        cumsum += angle;
    }
    // normalize the steps so that point 0 and point n+1 are the same
    cumsum /= (2. * PI);
    for i in 0..steps {
        angles[i] /= cumsum;
    }
    angles
}


pub fn clip(value: f64, lower: f64, upper: f64) -> f64 {
    return upper.min(value.max(lower))
}

use dyn_clone::DynClone;
use serde_derive::{Deserialize, Serialize};
use crate::common::{build_complex_vector, matrix_to_vec};
use crate::initial;
use crate::stream::matrix_to_file;
use crate::tasks::TaskParameters;


pub trait Figure {

    fn is_belongs(&self, x: f64, y: f64) -> bool;

    fn draw(&self, x: f64, y: f64) -> f64;
}


pub struct Polygon {
    pub points: Vec<(f64, f64)>,
    pub k: f64,
}

impl Polygon {
    pub fn new(points: Vec<(f64, f64)>, k: f64) -> Self {
        Self {
            points,
            k,
        }
    }
}


impl Figure for Polygon {
    fn is_belongs(&self, x: f64, y: f64) -> bool {
        let (mut min_x, mut min_y) = (self.points[0].0, self.points[0].1);
        let (mut max_x, mut max_y) = (self.points[0].0, self.points[0].1);

        for i in 1..self.points.len() {
            min_x = self.points[i].0.min(min_x);
            max_x = self.points[i].0.max(max_x);
            min_y = self.points[i].1.min(min_y);
            max_y = self.points[i].1.max(max_y);
        }

        if x < min_x || x > max_x || y < min_y || y > max_y {
            return false;
        }

        let mut inside = false;
        let mut j = self.points.len() - 1;
        for i in 0..self.points.len() {
            if ((self.points[i].1 > y) != (self.points[j].1 > y)) &&
                (x < (self.points[j].0 - self.points[i].0) * (y - self.points[i].1) / (
                    self.points[j].1 - self.points[i].1) +
                    self.points[i].0) {
                inside = !inside;
            }

            j = i
        }
        inside

    }

    fn draw(&self, x: f64, y: f64) -> f64 {

        match self.is_belongs(x, y) {
            true => {
                self.k
            }
            false => {
                0.0
            }
        }
    }
}

pub struct Surface {
    pub rows: usize,
    pub cols: usize,
    pub cell_size: usize,
    pub k0: f64,
    pub figures: Vec<Box<dyn Figure>>
}

impl Surface {

    pub fn new(height: usize, width: usize, cell_size: usize, k0: f64) -> Self {
        Self {
            rows: height * cell_size,
            cols: width * cell_size,
            cell_size,
            k0,
            figures: vec![],
        }
    }

    pub fn height(&self) -> usize {
        self.rows / self.cell_size
    }

    pub fn width(&self) -> usize {
        self.cols / self.cell_size
    }

    pub fn add_figure<F: Figure + 'static>(&mut self, figure: F) {
        self.figures.push(Box::new(figure));
    }

    pub fn set_k0(mut self, k0: f64) -> Self {
        self.k0 = k0;
        self
    }

    pub fn get_k_matrix(&self) -> Vec<Vec<f64>>{
        let mut matrix = vec![vec![0.0; self.cols]; self.rows];
        for i in 0..self.rows {
            let y = i / self.cell_size;
            for j in 0..self.cols {
                let x = j / self.cell_size;
                for fig in self.figures.iter() {
                    let val = fig.draw(x as f64, y as f64);
                    matrix[i][j] = if val > matrix[i][j] {val} else {matrix[i][j]};
                }
                if matrix[i][j] == 0.0 {
                    matrix[i][j] = self.k0;
                }
            }
        }
        matrix
    }

    pub fn get_w_matrix(&self) -> Vec<Vec<f64>>{
        let mut matrix = vec![vec![0.0; self.cols]; self.rows];
        for i in 0..self.rows {
            let y = i / self.cell_size;
            for j in 0..self.cols {
                let x = j / self.cell_size;
                for fig in self.figures.iter() {
                    let val = fig.draw(x as f64, y as f64);
                    matrix[i][j] = if val > matrix[i][j] {1.0} else {matrix[i][j]};
                }

            }
        }
        matrix
    }

}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolygonCoveringResult {
    pub center: (f64, f64),
    pub points: Vec<(f64, f64)>,
}

impl PolygonCoveringResult {
    pub fn save_to_file(&self, file_path: &Path) -> std::io::Result<()> {
        let serialized = serde_json::to_string_pretty(&self)?;
        std::fs::write(file_path, serialized)?;
        Ok(())
    }
}

pub fn polygon_covering(surface: &mut Surface, each_size_pct: f64, each_proba: f64, limit: usize,
                        k0_dev: f64,
                        irregularity: f64, spikiness: f64, num_vertices: usize) -> anyhow::Result<Vec<PolygonCoveringResult>> {

    let mut res = vec![];

    let each_abs_size = (each_size_pct * surface.rows as f64, each_size_pct * surface.cols as f64);
    let mut counter = 0;
    let mut rng = rand::thread_rng();
    let born_gen = Uniform::from(0.0..1.0);

    let k_gen = Uniform::from(surface.k0..surface.k0 + surface.k0 * k0_dev);

    let (offset_i, offset_j) = (each_abs_size.0.round() as usize * 1, each_abs_size.1.round() as usize * 1);
    let (step_i, step_j) = (each_abs_size.0.round() as usize * 4, each_abs_size.1.round() as usize * 4);

    for i in (offset_i..surface.rows - offset_i).step_by(step_i) {
        for j in (0..surface.cols - offset_j).step_by(step_j) {

            if counter >= limit {
                break
            }

            if born_gen.sample(&mut rng) > each_proba {
                continue
            }

            let center = (i as f64 + each_abs_size.0, j as f64 + each_abs_size.1);
            let rad = (each_abs_size.0.powi(2) + each_abs_size.1.powi(2)).sqrt();

            let points = generate_polygon(center, rad, irregularity, spikiness, num_vertices)?;
            let p = Polygon::new(points.clone(), k_gen.sample(&mut rng));
            surface.add_figure(p);

            counter += 1;

            res.push(PolygonCoveringResult {
                center,
                points,
            })
        }
    }

    Ok(res)
}
