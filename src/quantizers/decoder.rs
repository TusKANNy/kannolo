pub trait Decoder {
    fn decode(&mut self) -> usize;
}

pub struct PQDecoder8<'a> {
    code: &'a [u8],
    position: usize,
}

impl<'a> PQDecoder8<'a> {
    pub fn new(code: &'a [u8]) -> Self {
        PQDecoder8 { code, position: 0 }
    }

    #[inline]
    fn decode_8(&mut self) -> usize {
        let result = self.code[self.position] as usize;
        self.position += 1;
        result
    }
}

impl<'a> Decoder for PQDecoder8<'a> {
    fn decode(&mut self) -> usize {
        self.decode_8()
    }
}

impl<'a, T> Decoder for &'a mut T
where
    T: Decoder + ?Sized,
{
    fn decode(&mut self) -> usize {
        (**self).decode()
    }
}

impl<T> Decoder for Box<T>
where
    T: Decoder + ?Sized,
{
    fn decode(&mut self) -> usize {
        (**self).decode()
    }
}
