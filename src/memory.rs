
pub fn create_vector_memory<T>(n: usize, val: T) -> Vec<T> where T: Clone {
    vec![val; n]
}

pub fn create_matrix_memory<T>(n: usize, val: T) -> Vec<Vec<T>> where T: Clone {
    vec![vec![val; n]; n]
}