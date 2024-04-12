fn main() {
    make_grid(2, 2, 0.15, 0.15);
}


fn abstract_points(p: usize, point: usize, dim_x: f64, dim_y: f64, a: f64, b: f64) -> Vec<(f64, f64)> {

    let len_x = dim_x / p as f64;
    let len_y = dim_y / p as f64;

    let mut points = vec![];

    let n = p * point;
    let l_x = len_x / point as f64;
    let l_y = len_x / point as f64;


    for i in 0..p {
        for j in 0..p {

            for ii in 0..point {
                for jj in 0..point {
                    let x_c = a + j as f64 * len_x + jj as f64 * l_x + l_x / 2.0;
                    let y_c = b + i as f64 * len_x + ii as f64 * l_y + l_y / 2.0;

                    points.push((x_c, y_c));

                }
            }

        }
    }

    points

}


fn make_grid(p: usize, point: usize, dim_x: f64, dim_y: f64) {


    let col_points = abstract_points(p, point, dim_x, dim_y, -dim_x / 2., -dim_y / 2.);
    let pos_points = get_pos_points(p, point, dim_x, dim_y, col_points.as_slice());
    let int_points = get_int_points(p, 3, dim_x, dim_y, pos_points.as_slice());

    println!("col = {:?}", col_points);
    println!();
    println!("pos = {:?}", pos_points);
    println!();
    println!("integ = {:?}", int_points);

}

fn get_int_points(p: usize, ip: usize, dim_x: f64, dim_y: f64, pos_points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let len_x = dim_x / p as f64;
    let len_y = dim_y / p as f64;

    let mut int_points = vec![];

    for (k, (pos_x, pos_y)) in pos_points.iter().enumerate() {
        // if k != 0 {
        //     continue
        // }
        println!("POS POINT{:?}", (pos_x, pos_y));
        for i in 0..ip {
            for j in 0..ip {

                int_points.push((pos_x + (len_x / ip as f64) * (j as f64 + 0.5), pos_y + (len_y / ip as f64) * (i as f64 + 0.5)))
            }
        }
    }
    // let int_points = abstract_points(p, ip, dim_x, dim_y, -dim_x / 2. - len_x / 2.0, -dim_y / 2. + len_y / 2.0);
    println!("{:?}", int_points.len());
    println!("{:?}", int_points[0]);
    println!("{:?}", pos_points[0]);
    int_points
}

fn get_pos_points(p: usize, point: usize, dim_x: f64, dim_y: f64, col_points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let len_x = dim_x / p as f64;
    let len_y = dim_y / p as f64;

    let mut pos_points: Vec<(f64, f64)> = col_points.iter()
        .map(|(x_col, y_col)| (x_col - len_x / 2.0, y_col - len_y / 2.0))
        .collect();
    pos_points
}
