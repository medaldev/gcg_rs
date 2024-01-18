use num::complex::Complex64;
use num::Zero;
use crate::matrix_system::fill_xy_col;
use crate::memory::create_vector_memory;

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

            if x * x + y * y <= r1 * r1 {
                K[ind] = Complex64::new(0.7, 0.);
                if (x < 0.){
                    K[ind] = Complex64::new(0.6, 0.);
                }

                W[ind] = Complex64::new(1.0, 0.);

                if (x * x + y * y <= r2 * r2) {
                    K[ind] = k1;
                    W[ind] = Complex64::zero();

                    if (x * x + y * y <= r3 * r3) {
                        K[ind] = Complex64::new(0.35, 0.);
                        if (x < 0.){
                            K[ind] = Complex64::new(0.25, 0.);
                        }

                        W[ind] = Complex64::new(1.0, 0.);
                        if (x * x + y * y <= r4 * r4) {
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