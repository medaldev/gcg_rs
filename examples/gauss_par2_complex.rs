use num::complex::Complex64;
use num::Zero;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::ParallelIterator;



fn main() {
    let A = vec![
        vec![Complex64::new(2., 0.), Complex64::new(4., 0.), Complex64::new(1., 0.)],
        vec![Complex64::new(5., 0.), Complex64::new(2., 0.), Complex64::new(1., 0.)],
        vec![Complex64::new(2., 0.), Complex64::new(3., 0.), Complex64::new(4., 0.)]
    ];

    let B = vec![Complex64::new(36., 0.), Complex64::new(47., 0.), Complex64::new(37., 0.)];

    let out = gauss_par(&A, &B);

    println!("{:?}", out);
}

type Matrix = Vec<Vec<Complex64>>;


pub fn gauss_par(A: &Matrix, B: &Vec<Complex64>) -> Vec<Complex64> {
    let mut AA = A.clone();
    for (row, el) in AA.iter_mut().zip(B.iter()) {
        row.push(el.clone());
    }
    gaussian_elimination(&mut AA);
    back_substitution(&mut AA)
}


fn gaussian_elimination(matrix: &mut Matrix)  {

    let n = matrix.len();

    for i in 0..n {
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


}

fn back_substitution(matrix: &mut Matrix) -> Vec<Complex64> {
    let n = matrix.len();
    let mut solution = vec![Complex64::zero(); n];

    for i in (0..n).rev() {
        solution[i] = matrix[i][n] / matrix[i][i];
        for k in 0..i {
            let u =  matrix[k][i];
            matrix[k][n] -=  u * solution[i];
        }
    }

    solution
}