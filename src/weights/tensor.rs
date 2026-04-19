/// A loaded weight tensor with raw byte data.
#[derive(Debug, Clone)]
pub struct WeightTensor {
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
    pub elem_bytes: usize,
}

impl WeightTensor {
    /// Total number of elements.
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// Access data as f32 slice (only valid if elem_bytes == 4).
    pub fn as_f32_slice(&self) -> &[f32] {
        assert_eq!(self.elem_bytes, 4);
        let ptr = self.data.as_ptr() as *const f32;
        let len = self.data.len() / 4;
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }
}
