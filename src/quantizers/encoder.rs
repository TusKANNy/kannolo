/// Represents a generic encoder functionality.
/// It is designed for product quantization, allowing for efficient encoding of high-dimensional vectors.
///
/// # Methods
///
/// * `encode` - Encodes the given data using the implemented encoding strategy.
///
pub trait Encoder {
    /// Encodes the given usize value.
    ///
    /// # Arguments
    ///
    /// * `x` - The value to be encoded.
    ///
    fn encode(&mut self, x: usize);
}

/// Represents a generic encoder for product quantization.
/// This encoder can handle a variable number of bits for encoding, up to a maximum of 64 bits.
///
/// # Fields
///
/// * `code` - A mutable slice of bytes representing the encoded data.
/// * `position` - The current position in the `code` slice.
/// * `offset` - The current bit offset in the current byte.
/// * `nbits` - The number of bits used for encoding each value.
/// * `reg` - A 64-bit register used for intermediate storage during encoding.
///
pub struct PQEncoderGeneric<'a> {
    code: &'a mut [u8],
    position: usize,
    offset: u8,
    nbits: usize,
    reg: u64,
}

impl<'a> PQEncoderGeneric<'a> {
    /// Constructs a new generic encoder for product quantization.
    ///
    /// # Arguments
    ///
    /// * `code` - A mutable slice of bytes to store the encoded data.
    /// * `nbits` - The number of bits to use for encoding each value.
    ///
    /// # Panics
    ///
    /// Panics if `nbits` is greater than 64.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::encoder::PQEncoderGeneric;
    ///
    /// let mut buffer = vec![0; 10];
    /// let encoder = PQEncoderGeneric::new(&mut buffer, 32);
    /// ```
    pub fn new(code: &'a mut [u8], nbits: usize) -> Self {
        assert!(nbits <= 64);
        Self {
            code,
            position: 0,
            offset: 0,
            nbits,
            reg: 0,
        }
    }

    /// Encodes a 64-bit value into a byte stream using the specified number of bits.
    ///
    /// # Workflow
    ///
    /// 1. **Bitwise Positioning**: The function starts by shifting the input value `x` to the left by
    ///    the current `offset`. This aligns the bits of `x` to their correct position in the ongoing
    ///    encoded stream. The aligned bits are then merged into `reg` using a bitwise OR operation.
    /// 2. **Handling Byte Overflow**: If `offset + nbits` exceeds 8 (the size of a byte), the function
    ///    splits the value across multiple bytes. It first stores the lower 8 bits of `reg` in the
    ///    current `code` position and shifts `x` right by 8 - `offset` to handle the next byte.
    /// 3. **Storing Remaining Bits**: The function then enters a loop to store the rest of the bits.
    ///    In each iteration, it stores the next 8 bits of `x` and updates `position`. This continues
    ///    until all bits of `x` are stored.
    /// 4. **Updating State**: Finally, the `offset` is updated to reflect the new bit position within
    ///    the current byte, and `reg` is updated with any remaining bits of `x`.
    ///
    /// # Arguments
    ///
    /// * `x` - The 64-bit value to be encoded.
    ///
    #[inline]
    fn encode_64(&mut self, x: u64) {
        self.reg |= x << self.offset;
        let mut x = x >> (8 - self.offset);

        if self.offset + self.nbits as u8 >= 8 {
            if self.position < self.code.len() {
                self.code[self.position] = self.reg as u8;
                self.position += 1;
            }

            for _ in 0..(self.nbits - (8 - self.offset as usize)) / 8 {
                if self.position < self.code.len() {
                    self.code[self.position] = x as u8;
                    self.position += 1;
                }
                x >>= 8;
            }

            self.offset = ((self.offset as usize + self.nbits) % 8) as u8;
            self.reg = x;
        } else {
            self.offset += self.nbits as u8;
        }
    }
}

impl<'a> Encoder for PQEncoderGeneric<'a> {
    fn encode(&mut self, x: usize) {
        self.encode_64(x as u64);
    }
}

/// Represents an 8-bit encoder for product quantization.
/// This encoder is specialized for 8-bit encoding.
///
/// # Fields
///
/// * `code` - A mutable slice of bytes representing the encoded data.
/// * `position` - The current position in the `code` slice.
///
pub struct PQEncoder8<'a> {
    code: &'a mut [u8],
    position: usize,
}

impl<'a> PQEncoder8<'a> {
    /// Constructs a new 8-bit encoder for product quantization.
    ///
    /// # Arguments
    ///
    /// * `code` - A mutable slice of bytes to store the encoded data.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::encoder::PQEncoder8;
    ///
    /// let mut buffer = vec![0; 10];
    /// let encoder = PQEncoder8::new(&mut buffer);
    /// ```
    pub fn new(code: &'a mut [u8]) -> Self {
        PQEncoder8 { code, position: 0 }
    }

    /// Encodes an 8-bit value into a byte stream.
    ///
    /// # Workflow
    ///
    /// The encoding process for 8-bit values is straightforward. The function simply stores the
    /// 8-bit value `x` directly into the next position in the `code` array and then increments
    /// the `position`. This approach is efficient as each value directly fits within a single byte.
    ///
    /// # Arguments
    ///
    /// * `x` - The 8-bit value to be encoded.
    ///
    #[inline]
    fn encode_8(&mut self, x: u8) {
        self.code[self.position] = x;
        self.position += 1;
    }
}

impl<'a> Encoder for PQEncoder8<'a> {
    fn encode(&mut self, x: usize) {
        self.encode_8(x as u8);
    }
}

/// Represents a 16-bit encoder for product quantization.
/// This encoder is specialized for 16-bit encoding.
///
/// # Fields
///
/// * `code` - A mutable slice of 16-bit integers representing the encoded data.
/// * `position` - The current position in the `code` slice.
///
pub struct PQEncoder16<'a> {
    code: &'a mut [u16],
    position: usize,
}

impl<'a> PQEncoder16<'a> {
    /// Constructs a new 16-bit encoder for product quantization.
    ///
    /// # Arguments
    ///
    /// * `code` - A mutable slice of bytes to store the encoded data.
    ///
    /// # Safety
    ///
    /// This function is unsafe as it performs a type cast from a slice of u8 to a slice of u16,
    /// which requires the original slice to be properly aligned and sized for u16 values.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::encoder::PQEncoder16;
    ///
    /// let mut buffer = vec![0; 10];
    /// let encoder = PQEncoder16::new(&mut buffer);
    /// ```
    pub fn new(code: &'a mut [u8]) -> Self {
        let code_16 = unsafe { &mut *(code as *mut [u8] as *mut [u16]) };
        PQEncoder16 {
            code: code_16,
            position: 0,
        }
    }

    /// Encodes a 16-bit value into a byte stream.
    ///
    /// # Workflow
    ///
    /// Similar to `encode_8`, the `encode_16` function encodes each 16-bit value `x` directly
    /// into the `code` array. It ensures that the `code` array is treated as an array of 16-bit
    /// integers, thereby accommodating the 16-bit value in a single position. After storing the value,
    /// it increments the `position` to point to the next slot in the array.
    ///
    /// # Arguments
    ///
    /// * `x` - The 16-bit value to be encoded.
    ///
    #[inline]
    fn encode_16(&mut self, x: u16) {
        self.code[self.position] = x;
        self.position += 1;
    }
}

impl<'a> Encoder for PQEncoder16<'a> {
    fn encode(&mut self, x: usize) {
        self.encode_16(x as u16);
    }
}

impl<'a, T> Encoder for &'a mut T
where
    T: Encoder + ?Sized,
{
    fn encode(&mut self, x: usize) {
        (**self).encode(x);
    }
}

impl<T> Encoder for Box<T>
where
    T: Encoder + ?Sized,
{
    fn encode(&mut self, x: usize) {
        (**self).encode(x);
    }
}
