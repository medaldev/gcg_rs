
#cargo run --release --example direct_pure -- --path_save ./output/real --p 67 --point 1
cargo run --release --example reduce_binary -- --path_from ./output/135 --path_save ./output/reduced/ --width 135 --height 135 --start 1 --step 2
cargo run --release --example difference_uvych -- --path_first ./output/reduced --path_second ./output/real --path_save ./output/diff --name Uvych --p 67 --point 1
python plotter.py ./output/diff/Uvych_diff_abs.xls &