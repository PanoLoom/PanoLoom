//! Feature-gated parallel map helpers. With `parallel` (default; also the
//! wasm mt build) these fan out over rayon; without it they run inline.
//! Both variants produce results in input order, so outputs are identical
//! and every parity test covers both configurations by construction.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Map `f` over `items`, preserving order.
#[cfg(feature = "parallel")]
pub(crate) fn map<T, U, F>(items: &[T], f: F) -> Vec<U>
where
    T: Sync,
    U: Send,
    F: Fn(&T) -> U + Sync + Send,
{
    items.par_iter().map(f).collect()
}

#[cfg(not(feature = "parallel"))]
pub(crate) fn map<T, U, F>(items: &[T], f: F) -> Vec<U>
where
    F: Fn(&T) -> U,
{
    items.iter().map(f).collect()
}

/// Run `f` over each mutable row chunk of `data` (chunks of `chunk_len`),
/// passing the chunk index. Chunks are disjoint, so this is order-free.
#[cfg(feature = "parallel")]
pub(crate) fn for_each_chunk_mut<T, F>(data: &mut [T], chunk_len: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync + Send,
{
    data.par_chunks_mut(chunk_len)
        .enumerate()
        .for_each(|(i, c)| f(i, c));
}

#[cfg(not(feature = "parallel"))]
pub(crate) fn for_each_chunk_mut<T, F>(data: &mut [T], chunk_len: usize, f: F)
where
    F: Fn(usize, &mut [T]),
{
    data.chunks_mut(chunk_len)
        .enumerate()
        .for_each(|(i, c)| f(i, c));
}
