use crate::{DType, OmfModel, Tensor};
use half::f16;
use std::io::{self, Write};

const MAGIC: &[u8; 3] = b"OMF";
const VERSION: u16 = 1;

fn put_section<W: Write>(out: &mut W, id: u8, data: &[u8]) -> io::Result<()> {
    out.write_all(&[id])?;
    out.write_all(&(data.len() as u64).to_le_bytes())?;
    out.write_all(data)
}

fn encode_tensor(tensor: &Tensor, out: &mut Vec<u8>) -> io::Result<()> {
    let name = tensor.name.as_bytes();
    let name_len = u16::try_from(name.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "tensor name too long"))?;
    out.extend_from_slice(&name_len.to_le_bytes());
    out.extend_from_slice(name);
    out.push(tensor.dtype as u8);
    let rank = u8::try_from(tensor.shape.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "tensor rank too large"))?;
    out.push(rank);
    for dim in &tensor.shape {
        out.extend_from_slice(&dim.to_le_bytes());
    }
    let expected = tensor.shape.iter().try_fold(1usize, |n, &d| {
        n.checked_mul(usize::try_from(d).ok()?)
    }).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "invalid tensor shape"))?;
    if expected != tensor.values.len() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "tensor shape/value mismatch"));
    }

    let mut data = Vec::new();
    match tensor.dtype {
        DType::Fp16 => {
            for &value in &tensor.values {
                data.extend_from_slice(&f16::from_f32(value).to_bits().to_le_bytes());
            }
        }
        DType::Q4_0 | DType::Q8_0 => {
            for block in tensor.values.chunks(32) {
                let max = block.iter().fold(0.0f32, |m, x| m.max(x.abs()));
                let divisor = if tensor.dtype == DType::Q4_0 { 7.0 } else { 127.0 };
                let scale = if max == 0.0 { 0.0 } else { max / divisor };
                data.extend_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
                let mut quantized = [0i8; 32];
                if scale != 0.0 {
                    for (dst, &value) in quantized.iter_mut().zip(block) {
                        let lower = if tensor.dtype == DType::Q4_0 { -8.0 } else { -127.0 };
                        let upper = if tensor.dtype == DType::Q4_0 { 7.0 } else { 127.0 };
                        *dst = (value / scale).round().clamp(lower, upper) as i8;
                    }
                }
                if tensor.dtype == DType::Q4_0 {
                    for i in (0..32).step_by(2) {
                        let lo = (quantized[i] + 8) as u8;
                        let hi = (quantized[i + 1] + 8) as u8;
                        data.push(lo | (hi << 4));
                    }
                } else {
                    data.extend(quantized.iter().map(|x| *x as u8));
                }
            }
        }
    }
    out.extend_from_slice(&(data.len() as u64).to_le_bytes());
    out.extend_from_slice(&data);
    Ok(())
}

pub fn write_omf<W: Write>(mut out: W, model: &OmfModel) -> io::Result<()> {
    let params: u64 = model.tensors.values().map(|t| t.values.len() as u64).sum();
    out.write_all(MAGIC)?;
    out.write_all(&VERSION.to_le_bytes())?;
    out.write_all(&params.to_le_bytes())?;
    put_section(&mut out, 1, &serde_json::to_vec(&model.config).map_err(io::Error::other)?)?;
    put_section(&mut out, 2, &serde_json::to_vec(&model.tokenizer).map_err(io::Error::other)?)?;
    let count = u32::try_from(model.tensors.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "too many tensors"))?;
    let mut weights = count.to_le_bytes().to_vec();
    for tensor in model.tensors.values() {
        encode_tensor(tensor, &mut weights)?;
    }
    put_section(&mut out, 3, &weights)?;
    put_section(&mut out, 4, &serde_json::to_vec(&model.metadata).map_err(io::Error::other)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::omf_reader::read_omf;
    use serde_json::json;
    use std::collections::BTreeMap;

    #[test]
    fn round_trips_all_supported_dtypes() {
        let mut tensors = BTreeMap::new();
        for (name, dtype) in [("half", DType::Fp16), ("four", DType::Q4_0), ("eight", DType::Q8_0)] {
            tensors.insert(name.to_string(), Tensor {
                name: name.to_string(), shape: vec![3], dtype, values: vec![-1.0, 0.0, 1.0],
            });
        }
        let model = OmfModel {
            config: json!({"hidden_size": 1}), tokenizer: json!({"type": "byte"}),
            metadata: json!({}), tensors,
        };
        let mut bytes = Vec::new();
        write_omf(&mut bytes, &model).unwrap();
        let decoded = read_omf(bytes.as_slice()).unwrap();
        assert_eq!(decoded.tensors.len(), 3);
        assert!((decoded.tensors["half"].values[2] - 1.0).abs() < 0.001);
        assert_eq!(decoded.tensors["four"].dtype, DType::Q4_0);
        assert!((decoded.tensors["eight"].values[0] + 1.0).abs() < 0.01);
    }
}
