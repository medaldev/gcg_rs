use rayon::prelude::*;

fn main() {
    let mut A = vec![vec![2., 4., 1.], vec![5., 2., 1.], vec![2., 3., 4.]];
    let mut B = vec![36., 47., 37.];

    let m = gaussian_elimination(&mut A, &mut B);

    let out = back_substitution(&mut A, &mut B);
    println!("{:?}", A);
    println!("{:?}", out);
}

fn gaussian_elimination(a: &mut Vec<Vec<f64>>, b: &mut Vec<f64>) {
    let n = a.len();

    for i in 0..n {
        // Find the pivot for column i and swap rows in both A and B
        let max = (i..n).max_by(|&x, &y| a[x][i].partial_cmp(&a[y][i]).unwrap()).unwrap();
        a.swap(i, max);
        b.swap(i, max);

        // Perform elimination on lower rows
        for j in (i + 1)..n {
            let factor = a[j][i] / a[i][i];
            for k in i..n {
                a[j][k] -= a[i][k] * factor;
            }
            b[j] -= b[i] * factor;
        }
    }
}

fn back_substitution(a: &Vec<Vec<f64>>, b: &Vec<f64>) -> Vec<f64> {
    let n = a.len();
    let mut solution = vec![0.0; n];

    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i][j] * solution[j];
        }
        solution[i] = sum / a[i][i];
    }

    solution
}