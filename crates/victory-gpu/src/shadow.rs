//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (system memory shaders may also write), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: write
//! watching, or read-only guards on physical-page views), all of it where
//! neither works. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! shortcut: only the ranges draws are known to read are brought up to
//! date (vertex and index buffer views, constant buffers in tables, root
//! descriptors, buffer views in tables); buffers read through directly
//! indexed heaps keep what their first use copied.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::watch::PAGE;

/// A title buffer's host copy.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// Per page of title memory from `start`'s on: the write-watch asking
    /// after which it was last copied (0: never).
    copied: Mutex<Vec<u64>>,
    /// Whether the memory is write-watched: unknown, yes, no.
    watched: AtomicU8,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<Vec<(usize, usize)>>,
    /// Persistent mirror of `self.data`, allocated once on the first shader
    /// write to this buffer and kept in sync on CPU copies and copy-backs so
    /// subsequent submissions avoid snapshotting the entire buffer.
    mirror: Mutex<Option<Vec<u8>>>,
}

const NOT_WATCHED: u8 = 2;

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// Whether this shadow overlaps the title range `[addr, addr + len)`.
    pub(crate) fn overlaps(&self, addr: usize, len: usize) -> bool {
        let start = self.start as usize;
        let end = start.saturating_add(self.len as usize);
        let req_end = addr.saturating_add(len);
        start < req_end && end > addr
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    fn new(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64) -> Shadow {
        let pages = (start + len).div_ceil(PAGE as u64) - start / PAGE as u64;
        Shadow {
            buffer,
            data,
            gpu,
            start,
            len,
            copied: Mutex::new(vec![0; pages as usize]),
            watched: AtomicU8::new(0),
            pending: Mutex::default(),
            mirror: Mutex::default(),
        }
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.left(address));
        list.defer_gpu_write(self, address as usize, len as usize);
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Ensures the persistent mirror of `self.data` is
    /// initialized once (subsequent submissions reuse it with zero full-buffer
    /// memcpy) and records the range in `pending`.
    pub(crate) fn writing(&self, address: usize, len: usize) {
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((address, address + len));
        let mut mirror = self.mirror.lock().unwrap_or_else(|e| e.into_inner());
        if mirror.is_none() {
            // SAFETY: inside the persistent mapping of `self.len` bytes.
            let host = unsafe { std::slice::from_raw_parts(self.data, self.len as usize) };
            *mirror = Some(host.to_vec());
        }
    }

    /// After the submission ran: merges all pending shader-written ranges for
    /// this shadow buffer into disjoint intervals, scans each byte at most
    /// once against the persistent mirror (in parallel for large ranges),
    /// copies only GPU-modified bytes back into title memory, and updates the
    /// mirror in place. Returns the bytes written.
    pub(crate) fn copy_back_pending(&self) -> usize {
        let pending_ranges: Vec<(usize, usize)> = {
            let mut p = self.pending.lock().unwrap_or_else(|e| e.into_inner());
            std::mem::take(&mut *p)
        };
        if pending_ranges.is_empty() {
            return 0;
        }
        let mut intervals: Vec<(usize, usize)> = pending_ranges
            .into_iter()
            .filter_map(|(a, e)| {
                let (from, to) = self.inside(a, e.saturating_sub(a));
                (from < to).then_some((from, to))
            })
            .collect();
        if intervals.is_empty() {
            return 0;
        }
        intervals.sort_unstable();
        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(intervals.len());
        for (from, to) in intervals {
            if let Some(last) = merged.last_mut()
                && from <= last.1
            {
                last.1 = last.1.max(to);
            } else {
                merged.push((from, to));
            }
        }

        let mut mirror_guard = self.mirror.lock().unwrap_or_else(|e| e.into_inner());
        let Some(mirror) = mirror_guard.as_mut() else {
            return 0;
        };
        const PIECE: usize = 64 << 10;
        let start = self.start as usize;

        // Collect all 64 KiB-aligned pieces across the merged ranges.
        let mut pieces: Vec<(usize, usize)> = Vec::new();
        for &(from, to) in &merged {
            let mut at = from;
            while at < to {
                let n = (PIECE - (start + at) % PIECE).min(to - at);
                pieces.push((at, n));
                at += n;
            }
        }

        // Find dirty pieces in parallel using a persistent worker pool (zero
        // per-submission OS thread creation overhead).
        let dirty: Vec<(usize, usize)> = if pieces.len() >= 16 {
            scan_dirty_parallel(self.data, mirror, &pieces)
        } else {
            pieces
                .into_iter()
                .filter(|&(at, n)| {
                    // SAFETY: inside the mapping.
                    let host = unsafe { std::slice::from_raw_parts(self.data.add(at), n) };
                    host != &mirror[at..at + n]
                })
                .collect()
        };

        if dirty.is_empty() {
            return 0;
        }

        let mut written = 0;
        let mut title = vec![0u8; PIECE];
        let mut synced = Vec::with_capacity(dirty.len());
        for (at, n) in dirty {
            // SAFETY: inside the mapping; only the submitting thread writes it.
            let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
            let old = &mut mirror[at..at + n];
            let title = &mut title[..n];
            if crate::host::read_into((start + at) as *const u8, title) {
                let mut changed = false;
                for ((t, h), o) in title.iter_mut().zip(host.iter_mut()).zip(old.iter_mut()) {
                    if *h != *o && *t != *h {
                        *t = *h;
                        changed = true;
                    }
                    *h = *t;
                    *o = *t;
                }
                if changed {
                    crate::watch::writing(start + at, n);
                    crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
                    written += n;
                }
                synced.push((start + at) / PAGE);
            } else {
                // Keep mirror synced even on unmapped holes.
                old.copy_from_slice(host);
            }
        }

        if !synced.is_empty() && self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
            // Query GetWriteWatch only on contiguous runs of dirty synced pages
            // instead of the entire merged buffer span (often 174 MB).
            synced.sort_unstable();
            synced.dedup();
            let base = start / PAGE;
            let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
            let mut i = 0;
            while i < synced.len() {
                let first = synced[i];
                let mut last = first + 1;
                i += 1;
                while i < synced.len() && synced[i] <= last + 4 {
                    last = synced[i] + 1;
                    i += 1;
                }
                if let Some((now, _)) = crate::watch::writes(first * PAGE, last * PAGE, true) {
                    for &page in synced.iter().filter(|&&p| (first..last).contains(&p)) {
                        if let Some(slot) = copied.get_mut(page - base) {
                            *slot = now;
                        }
                    }
                }
            }
        }
        written
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return 0;
        }
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        if !busy.is_empty() {
            // Around ranges shaders are writing: the gaps, each as below
            // (in the world most submissions have such ranges; copying the
            // gaps whole cost ~1 s per frame).
            busy.sort_unstable();
            let (mut at, mut bytes) = (from, 0);
            for (a, e) in busy {
                if a > at {
                    bytes += self.update_free(at, a);
                }
                at = at.max(e);
            }
            if at < to {
                bytes += self.update_free(at, to);
            }
            return bytes;
        }
        self.update_free(from, to)
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where no
    /// shader writes.
    fn update_free(&self, from: usize, to: usize) -> usize {
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
            if let Some((now, written)) = crate::watch::writes(from, to, true) {
                self.watched.store(1, Ordering::Relaxed);
                let copied = self.copy_stale(from, to, now, &written);
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    add(C::Checked, (to - from) as u64);
                    add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
                }
                return copied;
            }
            self.watched.store(NOT_WATCHED, Ordering::Relaxed);
        }
        self.copy(from, to);
        to - from
    }

    /// Copies the pages of `[from, to)` written after their last copy
    /// (`written`: last writes by page, in page order; `now`: this
    /// asking), whole pages within the buffer.
    fn copy_stale(&self, from: usize, to: usize, now: u64, written: &[(usize, u64)]) -> usize {
        let base = self.start as usize / PAGE;
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let mut writes = written.iter().peekable();
        let (mut bytes, mut run) = (0, None);
        let last = (to - 1) / PAGE;
        for page in from / PAGE..=last {
            let write = match writes.peek() {
                Some(&&(p, w)) if p == page => {
                    writes.next();
                    w
                }
                _ => 0,
            };
            let at = &mut copied[page - base];
            if *at == 0 || write > *at {
                // A page only partly in the range is copied in part: it
                // stays stale for the rest.
                let (start, end) = (self.start as usize, (self.start + self.len) as usize);
                if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                    *at = now;
                }
                run.get_or_insert(page);
            } else if let Some(first) = run.take() {
                bytes += self.copy((first * PAGE).max(from), (page * PAGE).min(to));
            }
        }
        // Cut to the range: next to it shaders may be writing the copy.
        if let Some(first) = run {
            bytes += self.copy((first * PAGE).max(from), ((last + 1) * PAGE).min(to));
        }
        bytes
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy
    /// (and keeps the persistent mirror in sync if initialized).
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        let offset = from - start;
        let len = to - from;
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(offset) };
        crate::host::copy_mapped(at, from as *const u8, len, false);
        if let Some(mirror) = self
            .mirror
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_mut()
        {
            // SAFETY: `at..at+len` is inside `self.data` and just written above.
            let updated = unsafe { std::slice::from_raw_parts(at, len) };
            mirror[offset..offset + len].copy_from_slice(updated);
        }
        len
    }
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
static BUFFERS: Mutex<BTreeMap<(u64, usize), Entry>> = Mutex::new(BTreeMap::new());

fn buffers() -> std::sync::MutexGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    buffers().insert(
        (start, resource),
        Entry {
            end: start + len,
            shadow: None,
        },
    );
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    buffers().remove(&(start, resource));
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use.
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    let mut map = buffers();
    let key = map
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end >= end)
        .map(|(k, _)| *k);
    let Some(key) = key else {
        drop(map);
        return loose(address, end);
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let (buffer, data) = host()?.shared_buffer(len.next_multiple_of(1 << 16))?;
        crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x}", key.0));
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer copied to the host"
            );
        }
        entry.shadow = Some(Arc::new(Shadow::new(buffer, data, gpu, key.0, len)));
    }
    entry.shadow.clone()
}

/// Titles also use memory outside any buffer object as GPU memory (Grand
/// Theft Auto VI's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches.
/// shortcut: copies of windows are never freed; key them to the title's
/// allocations if memory use grows.
const WINDOW: u64 = 4 << 20;

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: Mutex<BTreeMap<(u64, u64), Arc<Shadow>>> = Mutex::new(BTreeMap::new());

fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    let start = address & !(WINDOW - 1);
    let len = end.next_multiple_of(WINDOW) - start;
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(shadow) = map.get(&(start, len)) {
        return Some(shadow.clone());
    }
    let (buffer, data) = host()?.shared_buffer(len)?;
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let shadow = Arc::new(Shadow::new(buffer, data, gpu, start, len));
    map.insert((start, len), shadow.clone());
    Some(shadow)
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}

/// Persistent worker pool for parallel dirty-chunk scanning in `copy_back_pending`.
struct ScanJob {
    epoch: u64,
    data: usize,
    mirror: usize,
    pieces_ptr: usize,
    pieces_len: usize,
    results_ptr: usize,
    done: usize,
}

struct ScanPool {
    workers: usize,
    state: Mutex<ScanJob>,
    cv_work: std::sync::Condvar,
    cv_done: std::sync::Condvar,
}

fn scan_pool() -> &'static ScanPool {
    static POOL: std::sync::OnceLock<&'static ScanPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        let workers = std::thread::available_parallelism()
            .map(|n| n.get().clamp(2, 8))
            .unwrap_or(4);
        let pool: &'static ScanPool = Box::leak(Box::new(ScanPool {
            workers,
            state: Mutex::new(ScanJob {
                epoch: 0,
                data: 0,
                mirror: 0,
                pieces_ptr: 0,
                pieces_len: 0,
                results_ptr: 0,
                done: 0,
            }),
            cv_work: std::sync::Condvar::new(),
            cv_done: std::sync::Condvar::new(),
        }));
        for id in 0..workers {
            std::thread::Builder::new()
                .name(format!("victory-scan-{id}"))
                .spawn(move || {
                    let mut seen_epoch = 0u64;
                    loop {
                        let (data, mirror, pieces_ptr, pieces_len, results_ptr, epoch) = {
                            let mut st = pool.state.lock().unwrap_or_else(|e| e.into_inner());
                            while st.epoch == seen_epoch {
                                st = pool.cv_work.wait(st).unwrap_or_else(|e| e.into_inner());
                            }
                            (
                                st.data,
                                st.mirror,
                                st.pieces_ptr,
                                st.pieces_len,
                                st.results_ptr,
                                st.epoch,
                            )
                        };
                        seen_epoch = epoch;
                        let chunk = pieces_len.div_ceil(pool.workers);
                        let lo = (id * chunk).min(pieces_len);
                        let hi = ((id + 1) * chunk).min(pieces_len);
                        let mut local = Vec::new();
                        if lo < hi {
                            // SAFETY: caller blocks on `cv_done` until all workers finish,
                            // keeping `data`, `mirror`, `pieces`, and `results[id]` alive.
                            let pieces = unsafe {
                                std::slice::from_raw_parts(
                                    pieces_ptr as *const (usize, usize),
                                    pieces_len,
                                )
                            };
                            let data_ptr = data as *const u8;
                            let mirror_ptr = mirror as *const u8;
                            for &(at, n) in &pieces[lo..hi] {
                                // SAFETY: within mapped host buffer.
                                let host =
                                    unsafe { std::slice::from_raw_parts(data_ptr.add(at), n) };
                                // SAFETY: within mirror Vec of identical length.
                                let old =
                                    unsafe { std::slice::from_raw_parts(mirror_ptr.add(at), n) };
                                if host != old {
                                    local.push((at, n));
                                }
                            }
                            // SAFETY: each worker owns its exclusive slot `results[id]`.
                            unsafe {
                                *(results_ptr as *mut Vec<(usize, usize)>).add(id) = local;
                            }
                        }
                        let mut st = pool.state.lock().unwrap_or_else(|e| e.into_inner());
                        st.done += 1;
                        if st.done == pool.workers {
                            pool.cv_done.notify_one();
                        }
                    }
                })
                .expect("spawn victory-scan worker");
        }
        pool
    })
}

fn scan_dirty_parallel(
    data: *const u8,
    mirror: &[u8],
    pieces: &[(usize, usize)],
) -> Vec<(usize, usize)> {
    let pool = scan_pool();
    let mut results: Vec<Vec<(usize, usize)>> = vec![Vec::new(); pool.workers];
    {
        let mut st = pool.state.lock().unwrap_or_else(|e| e.into_inner());
        st.data = data as usize;
        st.mirror = mirror.as_ptr() as usize;
        st.pieces_ptr = pieces.as_ptr() as usize;
        st.pieces_len = pieces.len();
        st.results_ptr = results.as_mut_ptr() as usize;
        st.done = 0;
        st.epoch = st.epoch.wrapping_add(1);
        pool.cv_work.notify_all();
        while st.done < pool.workers {
            st = pool.cv_done.wait(st).unwrap_or_else(|e| e.into_inner());
        }
    }
    let total: usize = results.iter().map(Vec::len).sum();
    let mut out = Vec::with_capacity(total);
    for r in results {
        out.extend(r);
    }
    out
}
