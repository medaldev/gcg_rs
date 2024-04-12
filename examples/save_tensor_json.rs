use serde::{Serialize, Deserialize};
use serde_json;


fn main() {
    // Create a tensor
    let tensor = vec![
        vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ],
        vec![
            vec![7.0, 8.0, 9.0],
            vec![10.0, 11.0, 12.0],
        ],
    ];

    // Serialize the tensor to JSON and write it to a file
    let tensor_json = serde_json::to_string(&tensor).unwrap();
    std::fs::write("tensor.json", tensor_json).expect("Unable to write file");
}