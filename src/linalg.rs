use rayon::prelude::*;
use pbr::ProgressBar;



/*
void BuildMatrix(int n, mycomplex **A, mycomplex *W, mycomplex *B)
{
	int i, j;

	for (i = 0; i<n; i++) {
		if (W[i] == 0.0) {
			for (j = 0; j<n; j++) {
				A[i][j] = 0.0;
				A[j][i] = 0.0;
			}
			A[i][i] = 1.0;
			B[i] = 0.0;

		}
	}
}
 */


use num::complex::{Complex64, ComplexFloat};
use num::{Complex, Zero};

pub fn build_matrix(N: usize, A: &mut Vec<Vec<Complex64>>, W: &mut Vec<Complex64>, B: &mut Vec<Complex64>) {
    for i in 0..N {
        if W[i] == Complex64::zero() {
            for j in 0..N {
                A[i][j] = Complex64::zero();
                A[j][i] = Complex64::zero();
            }
            A[i][i] = Complex64::new(1.0, 0.0);
            B[i] = Complex64::zero();
        }

    }
}

fn is_equal_vector<T>(n: usize, a: &[T], b: &[T]) -> bool where T: Eq {
    for i in 0..n {
        if b[i] != a[i] {
            return false;
        };
    }
    return true;
}

fn is_equal_matrix<T>(n: usize, a: &Vec<Vec<T>>, b: &Vec<Vec<T>>) -> bool where T: Eq {
    for i in 0..n {
        for j in 0..n {
            if b[i][j] != a[i][j] {
                return false;
            };
        }
    }
    return true
}


pub fn min_max_f64_vec(vec: &Vec<f64>) -> (f64, f64) {
    let min = vec.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).cloned();
    let max = vec.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).cloned();
    (min.unwrap(), max.unwrap())
}



pub fn gauss2(n: usize, A: &Vec<Vec<Complex64>>, B: &Vec<Complex64>, W: &Vec<Complex64>, U: &mut Vec<Complex64>) -> Complex64 {
    let mut pb = ProgressBar::new(n as u64);
    pb.set_width(Some(75));
    let mut AA = A.clone();
    let mut BB = B.clone();
    let eps = 1e-19;
    let mut cc: Complex64;
    let mut bb: Complex64;
    let mut d: Complex64;
    let mut s = Complex64::zero();
    let mut mult = Complex64::zero();

    for k in 0..n {
        pb.inc();
        // if k % 100 == 0 {
        //     println!("gauss {}\t{}", n, k);
        // }
        if AA[k][k].abs() < eps {
            let mut kk = 0;
            for ii in k + 1..n {
                if (AA[ii][k]).abs() > eps && kk == 0 {
                    kk = 1;
                    cc = BB[k];
                    BB[k] = BB[ii];
                    BB[ii] = cc;

                    for jj in k..n {
                        bb = AA[k][jj];
                        AA[k][jj] = AA[ii][jj];
                        AA[ii][jj] = bb;
                    }
                }
            }
            if kk == 0 {
                println!("System error!!");
                std::process::exit(1);
            }
        }

        for j in k + 1..n {
            d = AA[j][k] / AA[k][k];
            for i in k..n {
                AA[j][i] = AA[j][i] - d * AA[k][i];
            }
            BB[j] = BB[j] - d * BB[k];
        }
    }



    mult = Complex::new(1., 0.);
    for ii in 0..n {
        mult *= AA[ii][ii];
    }

    for k in (0..n).rev() {
        d = Complex64::zero();
        for j in k..n {
            s = AA[k][j] * U[j];
            d = d + s;
        }
        U[k] = (BB[k] - d) / AA[k][k];

    }
    pb.finish_println("");
    mult

}


type Matrix = Vec<Vec<Complex64>>;
pub fn gauss(n: usize, A: &Matrix, B: &Vec<Complex64>,W: &Vec<Complex64>, U: &mut Vec<Complex64>)  {
    println!("Use Gauss elimination parallel.");
    let mut AA = A.clone();
    for (row, el) in AA.iter_mut().zip(B.iter()) {
        row.push(el.clone());
    }
    gaussian_elimination(&mut AA);
    back_substitution(&mut AA,  U)
}


fn gaussian_elimination(matrix: &mut Matrix)  {

    let n = matrix.len();

    let mut pb = ProgressBar::new(n as u64);
    pb.set_width(Some(75));

    for i in 0..n {
        pb.inc();
        // Find the pivot for column i
        let max = (i..n).max_by(|&x, &y| matrix[x][i].norm().partial_cmp(&matrix[y][i].norm()).unwrap()).unwrap();


        // Swap the rows if necessary
        matrix.swap(i, max);

        // Clone the pivot row elements that will be used in the elimination process.
        let pivot_row: Vec<Complex64> = matrix[i][i..].to_vec();

        matrix.par_iter_mut().enumerate().skip(i + 1).for_each(|(j, row)| {
            if j > i {
                let factor = row[i] / pivot_row[0];
                row.iter_mut().enumerate().skip(i).for_each(|(k, value)| {
                    if k >= i {
                        *value -= factor * pivot_row[k - i];
                    }
                });
            }
        });
    }
    pb.finish_println("");

}

fn back_substitution(matrix: &mut Matrix,  U: &mut Vec<Complex64>) {
    let n = matrix.len();

    for i in (0..n).rev() {
        U[i] = matrix[i][n] / matrix[i][i];
        for k in 0..i {
            let u =  matrix[k][i];
            matrix[k][n] -=  u * U[i];
        }
    }

}