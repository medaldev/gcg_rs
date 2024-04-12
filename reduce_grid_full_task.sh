
python forward_plot_one.py ./output/135/ &

# cargo run --release --example full_sol -- --path_save ./output/real --p 67 --point 1
python full_plot_one.py ./output/real/ &

./show_diff_uvych.sh
cargo run --release --example inverse_from_uvych -- --path_from ./output/reduced --path_save ./output/basic/ --p 67  --point 1
python full_plot_one.py ./output/basic/ &

./show_diff_uvych2.sh
cargo run --release --example inverse_from_uvych -- --path_from ./output/reduced --path_save ./output/basic/ --p 67  --point 1
python full_plot_one.py ./output/basic/ &