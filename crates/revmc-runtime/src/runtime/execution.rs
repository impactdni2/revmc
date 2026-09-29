//! Sample execution rather than lookups: cached dispatches and resumed frames still do work.

use super::{JitBackend, LookupDecision, stats::RuntimeStats};
use std::{
    cell::Cell,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

/// Each execution segment has probability 1/N of being timed, independent of its backend.
/// Counts and nanoseconds are raw samples, not extrapolated totals.
pub const EXECUTION_SAMPLE_RATE: u64 = 1024;
const SEED_STEP: u64 = 0x9e3779b97f4a7c15;
static NEXT_SEED: AtomicU64 = AtomicU64::new(SEED_STEP);

thread_local! {
    // Different streams prevent workers from sampling the same positions in repeated routes.
    static SAMPLE_STATE: Cell<u64> = Cell::new(NEXT_SEED.fetch_add(SEED_STEP, Ordering::Relaxed).max(1));
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum CallPhase {
    Start,
    Resume,
}

#[derive(Clone, Copy)]
pub(crate) enum ExecutionBackend {
    Compiled,
    Interpreter,
}

impl From<&LookupDecision> for ExecutionBackend {
    fn from(decision: &LookupDecision) -> Self {
        match decision {
            LookupDecision::Compiled(_) => Self::Compiled,
            LookupDecision::Interpret(_) => Self::Interpreter,
        }
    }
}

// Randomize positions to avoid periodic sampling aliasing with repeated call patterns.
#[inline]
fn should_sample() -> bool {
    #[cfg(test)]
    if SAMPLE_ALL.with(Cell::get) {
        return true;
    }

    SAMPLE_STATE.with(|state| {
        let mut next = state.get();
        next ^= next << 13;
        next ^= next >> 7;
        next ^= next << 17;
        state.set(next);
        next & (EXECUTION_SAMPLE_RATE - 1) == 0
    })
}

impl JitBackend {
    #[inline]
    pub(crate) fn sample_execution(
        &self,
        backend: ExecutionBackend,
        phase: CallPhase,
    ) -> Option<ExecutionSample> {
        if !should_sample() {
            return None;
        }

        // Only selected segments read the clock, clone the stats Arc, or update shared counters.
        let stats = Arc::clone(&self.inner.shared.stats);
        Some(ExecutionSample { stats, backend, phase, started: Instant::now() })
    }
}

pub(crate) struct ExecutionSample {
    stats: Arc<RuntimeStats>,
    backend: ExecutionBackend,
    phase: CallPhase,
    started: Instant,
}

impl Drop for ExecutionSample {
    fn drop(&mut self) {
        // Stop timing before counter updates. Drop also records error/revert paths.
        let elapsed = self.started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        let (calls, segments, nanos) = match self.backend {
            ExecutionBackend::Compiled => (
                &self.stats.sampled_compiled_calls,
                &self.stats.sampled_compiled_segments,
                &self.stats.sampled_compiled_ns,
            ),
            ExecutionBackend::Interpreter => (
                &self.stats.sampled_interpreter_calls,
                &self.stats.sampled_interpreter_segments,
                &self.stats.sampled_interpreter_ns,
            ),
        };
        segments.fetch_add(1, Ordering::Relaxed);
        nanos.fetch_add(elapsed, Ordering::Relaxed);
        if self.phase == CallPhase::Start {
            calls.fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_is_sparse() {
        assert!(EXECUTION_SAMPLE_RATE.is_power_of_two());
        SAMPLE_STATE.with(|state| state.set(SEED_STEP));
        let draws = EXECUTION_SAMPLE_RATE * 10_000;
        let samples = (0..draws).filter(|_| should_sample()).count();
        assert!((9_500..10_500).contains(&samples), "unexpected sample count: {samples}");
    }

    #[test]
    fn resumes_skip_call_counts() {
        let backend = JitBackend::disabled();
        for (kind, phase) in [
            (ExecutionBackend::Compiled, CallPhase::Start),
            (ExecutionBackend::Compiled, CallPhase::Resume),
            (ExecutionBackend::Interpreter, CallPhase::Start),
            (ExecutionBackend::Interpreter, CallPhase::Resume),
        ] {
            let sample = loop {
                if let Some(sample) = backend.sample_execution(kind, phase) {
                    break sample;
                }
            };
            drop(sample);
        }
        let stats = backend.stats();
        assert_eq!(stats.sampled_compiled_calls, 1);
        assert_eq!(stats.sampled_interpreter_calls, 1);
        assert_eq!(stats.sampled_compiled_segments, 2);
        assert_eq!(stats.sampled_interpreter_segments, 2);
        assert!(stats.sampled_compiled_ns > 0);
        assert!(stats.sampled_interpreter_ns > 0);
    }
}

#[cfg(test)]
thread_local! {
    static SAMPLE_ALL: Cell<bool> = const { Cell::new(false) };
}

#[cfg(test)]
pub(crate) fn with_all_samples<R>(run: impl FnOnce() -> R) -> R {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            SAMPLE_ALL.with(|state| state.set(self.0));
        }
    }
    let _reset = Reset(SAMPLE_ALL.with(|state| state.replace(true)));
    run()
}
