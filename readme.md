

```shell
cargo run --release \
--example reduce_binary -- \
--path_from ./experiments/output_p_40_point_3_ip3 \
--path_save ./input \
--width 120 --height 120 \
--start 1 --step 3
```


```shell
cargo run --release \
--example reduce_avg_binary -- \
--path_from ./experiments/output_p_40_point_3_ip3 \
--path_save ./input \
--width 120 --height 120 \
--window_width 3 --window_height 3 \
--stride_y 3 --stride_x 3
```

```shell
cargo run --release \
--example inverse_from_uvych -- \
--path_from ./experiments/output_p_20_point_2_ip3 \
--path_save ./output/trash/
```



