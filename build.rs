use std::process::Command;

fn main() {
    let output = Command::new("sh")
        .arg("-c")
        .arg("source ./setenv.sh && echo $LIBTORCH")
        .output()
        .expect("Failed to execute command");

    println!("{}", String::from_utf8(output.stdout).unwrap());
}