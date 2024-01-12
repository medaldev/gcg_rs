use num::complex::Complex64;
use num::Zero;
use crate::consts::{A, B, DIM_X, DIM_Y, K1, N, N1, N_X, N_Y, NUM_X, NUM_Y, POINT};
use crate::matrix_system::fill_xy_col;
use crate::memory::create_vector_memory;

pub fn initial_k0(n: usize, K: &mut [Complex64], W: &mut [Complex64]) {

    let (mut p1, mut p2, mut p3, mut p4) = (0, 0, 0, 0);
    let (mut x, mut y) = (0.0, 0.0);
    let mut s = Complex64::zero();

    let mut xc = create_vector_memory(n, 0.0f64);
    let mut yc = create_vector_memory(n, 0.0f64);

    fill_xy_col(n, NUM_X, NUM_Y, N_X, N_Y, DIM_X, DIM_Y, A, B, &mut xc, &mut yc);

    let r1 = DIM_X / 2.0;
    let r2 = 3.* DIM_X / 8.0;
    let r3 = DIM_X / 4.0;
    let r4 = DIM_X / 8.0;
    let mut ind = 0;

    for i in 0..N_X {
        for j in 0..N_Y {

            x = xc[N1 + ind];
            y = yc[N1 + ind];

            K[N1 + ind] = Complex64::new(0.7, 0.);
            if x < 0.0 {
                K[N1 + ind] = Complex64::new(0.6, 0.);
            }

            W[N1 + ind] = Complex64::new(1.0, 0.);

            ind += 1;
        }
    }

    ind = 0;
    for i in 0..N_X {
        for j in 0..N_Y {

            x = xc[N1 + ind];
            y = yc[N1 + ind];

            if x * x + y * y <= r1 * r1 {
                K[N1 + ind] = Complex64::new(0.7, 0.);
                if (x < 0.){
                    K[N1 + ind] = Complex64::new(0.6, 0.);
                }

                W[N1 + ind] = Complex64::new(1.0, 0.);

                if (x * x + y * y <= r2 * r2) {
                    K[N1 + ind] = K1;
                    W[N1 + ind] = Complex64::zero();

                    if (x * x + y * y <= r3 * r3) {
                        K[N1 + ind] = Complex64::new(0.35, 0.);
                        if (x < 0.){
                            K[N1 + ind] = Complex64::new(0.25, 0.);
                        }

                        W[N1 + ind] = Complex64::new(1.0, 0.);
                        if (x * x + y * y <= r4 * r4) {
                            K[N1 + ind] = K1;
                            W[N1 + ind] = Complex64::zero();
                        }

                    }
                }
            }
            else {
                K[N1 + ind] = K1;
                W[N1 + ind] = Complex64::zero();
            }

            ind += 1;
        }
    }

    ind = 0;
    for i2 in 0..NUM_Y {
        for i3 in 0..NUM_X {

            p1 = N1 + 2 * i2* N_X + 2 * i3;

            p2 = p1 + 1;
            p3 = p1 + N_X;
            p4 = p1 + N_X + 1;

            s = K[p1] + K[p2] + K[p3] + K[p4];
            K[ind] = s / (POINT * POINT) as f64;

            ind += 1;

        }
    }

    for i1 in 0..N {
        K[i1] *= W[i1];
    }

}