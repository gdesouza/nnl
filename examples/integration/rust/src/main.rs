extern "C" {
    fn simple_mlp_infer(input: *const f32, output: *mut f32) -> i32;
    fn simple_mlp_input_size() -> i32;
    fn simple_mlp_output_size() -> i32;
}

fn main() {
    let input_size = unsafe { simple_mlp_input_size() } as usize;
    let output_size = unsafe { simple_mlp_output_size() } as usize;

    println!("Input size:  {input_size}");
    println!("Output size: {output_size}");

    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let mut output = vec![0.0f32; output_size];

    let rc = unsafe { simple_mlp_infer(input.as_ptr(), output.as_mut_ptr()) };
    assert_eq!(rc, 0, "inference failed with code {rc}");

    println!("Output: {:?}", output);
    let predicted = output
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    println!("Predicted class: {predicted}");
}
