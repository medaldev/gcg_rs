use rayon::prelude::*;

fn main() {
    let mut A = vec![vec![2., 4., 1., 36.], vec![5., 2., 1., 47.], vec![2., 3., 4., 37.]];
    let B = vec![36., 47., 37.];

    let m = gaussian_elimination(&mut A);

    let out = back_substitution(&mut A);
    println!("{:?}", A);
    println!("{:?}", out);


}

type Matrix = Vec<Vec<f64>>;


fn gaussian_elimination(matrix: &mut Matrix)  {

    let n = matrix.len();

    for i in 0..n {
        // Find the pivot for column i
        let max = (i..n).max_by(|&x, &y| matrix[x][i].partial_cmp(&matrix[y][i]).unwrap()).unwrap();

        // Swap the rows if necessary
        matrix.swap(i, max);

        // Clone the pivot row elements that will be used in the elimination process.
        let pivot_row: Vec<f64> = matrix[i][i..].to_vec();

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

fn back_substitution(matrix: &mut Matrix) -> Vec<f64> {
    let n = matrix.len();
    let mut solution = vec![0.0; n];

    for i in (0..n).rev() {
        solution[i] = matrix[i][n] / matrix[i][i];
        for k in 0..i {
            matrix[k][n] -= matrix[k][i] * solution[i];
        }
    }

    solution
}