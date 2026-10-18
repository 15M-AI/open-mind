pub mod omf_reader;
pub mod omf_writer;

use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum DType {
    Fp16 = 1,
    Q4_0 = 2,
    Q8_0 = 3,
}

impl TryFrom<u8> for DType {
    type Error = String;
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Fp16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q8_0),
            _ => Err(format!("unsupported OMF dtype {value}")),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Tensor {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: DType,
    pub values: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct OmfModel {
    pub config: Value,
    pub tokenizer: Value,
    pub metadata: Value,
    pub tensors: BTreeMap<String, Tensor>,
}
