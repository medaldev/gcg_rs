cargo run --release
python full_plot_one.py ./output/ &
cargo run --release --example reduce_binary -- --path_from ./output/ --path_save ./output/reduced/ --width 61 --height 61 --start 1 --step 2
cargo run --release --example inverse_from_uvych -- --path_from ./output/reduced --path_save ./output/basic/ --p 30  --point 1
python full_plot_one.py ./output/basic/ &

cargo run --release --example reduce_avg_binary -- --path_from ./output/ --path_save ./output/reduced/ --width 61 --height 61 --window_width 2 --window_height 2 --stride_x 1 --stride_y 1
cargo run --release --example inverse_from_uvych -- --path_from ./output/reduced --path_save ./output/basic/ --p 60  --point 1
python full_plot_one.py ./output/basic/ &