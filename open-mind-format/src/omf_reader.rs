use crate::{DType, OmfModel, Tensor};
use half::f16;
use serde_json::Value;
use std::collections::BTreeMap;
use std::io::{self, Read};

const MAX_SECTION: u64 = 8 * 1024 * 1024 * 1024;
const MAX_TENSORS: u32 = 1_000_000;

fn read_u8<R: Read>(r: &mut R) -> io::Result<u8> {
    let mut b = [0; 1]; r.read_exact(&mut b)?; Ok(b[0])
}
fn read_u16<R: Read>(r: &mut R) -> io::Result<u16> {
    let mut b = [0; 2]; r.read_exact(&mut b)?; Ok(u16::from_le_bytes(b))
}
fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0; 4]; r.read_exact(&mut b)?; Ok(u32::from_le_bytes(b))
}
fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut b = [0; 8]; r.read_exact(&mut b)?; Ok(u64::from_le_bytes(b))
}
fn checked_len(value: u64) -> io::Result<usize> {
    if value > MAX_SECTION { return Err(io::Error::new(io::ErrorKind::InvalidData, "OMF section too large")); }
    usize::try_from(value).map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "OMF length overflow"))
}
fn json_section(bytes: &[u8]) -> io::Result<Value> {
    serde_json::from_slice(bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

pub fn read_omf<R: Read>(mut input: R) -> io::Result<OmfModel> {
    let mut magic = [0; 3]; input.read_exact(&mut magic)?;
    if &magic != b"OMF" { return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid OMF magic")); }
    let version = read_u16(&mut input)?;
    if version != 1 { return Err(io::Error::new(io::ErrorKind::InvalidData, format!("unsupported OMF version {version}"))); }
    let param_count = read_u64(&mut input)?;
    let mut config = None;
    let mut tokenizer = None;
    let mut metadata = None;
    let mut tensors = BTreeMap::new();
    let mut actual_params = 0u64;
    for expected_id in 1..=4 {
        let id = read_u8(&mut input)?;
        if id != expected_id { return Err(io::Error::new(io::ErrorKind::InvalidData, "missing, duplicate, or out-of-order OMF section")); }
        let len = checked_len(read_u64(&mut input)?)?;
        let mut payload = vec![0; len];
        input.read_exact(&mut payload)?;
        match id {
            1 => config = Some(json_section(&payload)?),
            2 => tokenizer = Some(json_section(&payload)?),
            3 => {
                let mut cursor = payload.as_slice();
                let count = read_u32(&mut cursor)?;
                if count > MAX_TENSORS { return Err(io::Error::new(io::ErrorKind::InvalidData, "too many tensors")); }
                for _ in 0..count {
                    let name_len = read_u16(&mut cursor)? as usize;
                    let mut name = vec![0; name_len]; cursor.read_exact(&mut name)?;
                    let name = String::from_utf8(name).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                    let dtype = DType::try_from(read_u8(&mut cursor)?).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                    let rank = read_u8(&mut cursor)? as usize;
                    let mut shape = Vec::with_capacity(rank);
                    let mut elements = 1usize;
                    for _ in 0..rank {
                        let d = read_u64(&mut cursor)?;
                        elements = elements.checked_mul(usize::try_from(d).map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "dimension overflow"))?)
                            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "tensor size overflow"))?;
                        shape.push(d);
                    }
                    let data_len = checked_len(read_u64(&mut cursor)?)?;
                    let expected_data_len = match dtype {
                        DType::Fp16 => elements.checked_mul(2).unwrap_or(usize::MAX),
                        DType::Q4_0 | DType::Q8_0 => {
                            let block_bytes = if dtype == DType::Q4_0 { 18 } else { 34 };
                            elements.saturating_add(31).checked_div(32)
                                .unwrap_or(usize::MAX).saturating_mul(block_bytes)
                        }
                    };
                    if data_len != expected_data_len {
                        return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid tensor data length"));
                    }
                    actual_params = actual_params.checked_add(elements as u64)
                        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "parameter count overflow"))?;
                    if actual_params > param_count {
                        return Err(io::Error::new(io::ErrorKind::InvalidData, "parameter count exceeds header"));
                    }
                    let mut data = vec![0; data_len]; cursor.read_exact(&mut data)?;
                    let mut values = Vec::with_capacity(elements);
                    match dtype {
                        DType::Fp16 => {
                            for b in data.chunks_exact(2) {
                                values.push(f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32());
                            }
                        }
                        DType::Q4_0 | DType::Q8_0 => {
                            let block_bytes = if dtype == DType::Q4_0 { 18 } else { 34 };
                            for (i, block) in data.chunks_exact(block_bytes).enumerate() {
                                let scale = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
                                for j in 0..32 {
                                    let q = if dtype == DType::Q4_0 {
                                        let byte = block[2 + j / 2];
                                        if j % 2 == 0 { (byte & 15) as i32 - 8 } else { (byte >> 4) as i32 - 8 }
                                    } else { (block[2 + j] as i8) as i32 };
                                    if i * 32 + j < elements { values.push(q as f32 * scale); }
                                }
                            }
                        }
                    }
                    if tensors.insert(name.clone(), Tensor { name, shape, dtype, values }).is_some() {
                        return Err(io::Error::new(io::ErrorKind::InvalidData, "duplicate OMF tensor"));
                    }
                }
                if !cursor.is_empty() { return Err(io::Error::new(io::ErrorKind::InvalidData, "trailing bytes in WEIGHTS")); }
            }
            4 => metadata = Some(json_section(&payload)?),
            _ => unreachable!(),
        }
    }
    if actual_params != param_count { return Err(io::Error::new(io::ErrorKind::InvalidData, "parameter count mismatch")); }
    Ok(OmfModel {
        config: config.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing CONFIG"))?,
        tokenizer: tokenizer.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing TOKENIZER"))?,
        metadata: metadata.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing METADATA"))?,
        tensors,
    })
}
