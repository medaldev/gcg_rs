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
use num::Zero;

pub fn build_matrix(N: usize, A: &mut Vec<Vec<Complex64>>, W: &mut Vec<Complex64>, B: &mut Vec<Complex64>) {
    for i in 0..N {
        if W[i] == Complex64::zero() {
            for j in 0..N {
                A[i][j] = Complex64::zero();
                A[j][i] = Complex64::zero();
            }
        }
        A[i][i] = Complex64::new(1.0, 0.0);
        B[i] = Complex64::zero();
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

pub fn gauss(n: usize, A: &Vec<Vec<Complex64>>, B: &Vec<Complex64>, W: &Vec<Complex64>, U: &mut Vec<Complex64>) -> Complex64 {
    let mut AA = A.clone();
    let mut BB = B.clone();
    let eps = 1e-19;
    let mut cc: Complex64;
    let mut bb: Complex64;
    let mut d: Complex64;
    let mut s = Complex64::zero();
    let mut mult = Complex64::zero();

    for k in 0..n {
        if k % 100 == 0 {
            println!("{}\t{}", n, k);
        }
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

    for ii in 0..n {
        mult *= AA[ii][ii];
    }

    for k in (0..n).rev() {
        d = Complex64::zero();
        for j in k..n {
            s += AA[k][j] * U[j];
        }
        U[k] = (BB[k] - d) / AA[k][k];
    }

    mult

}