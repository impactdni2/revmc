use crate::{
    EvmCompilerFn, eyre,
    runtime::{
        LookupRequest,
        api::{CompiledProgram, LoadedLibrary, ProgramKind},
        config::{
            ArtifactUsageEvent, CompilationEvent, CompilationKind, RuntimeConfig, RuntimeTuning,
        },
        storage::{
            ArtifactKey, ArtifactManifest, ArtifactStore, BackendSelection, RuntimeCacheKey,
            StoredArtifact,
        },
        worker::{
            AotSuccess, CompileJob, JitCodeBacking, JitObjectSuccess, SyncNotifier, WorkerPool,
            WorkerResult, WorkerSuccess,
        },
    },
};
use alloy_primitives::{
    Bytes, keccak256,
    map::{DefaultHashBuilder, HashMap},
};
use crossbeam_channel as chan;
use crossbeam_queue::ArrayQueue;
use dashmap::DashMap;
use quanta::Instant;
use std::{
    ffi::CString,
    mem,
    ops::ControlFlow,
    sync::{Arc, atomic::Ordering},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

#[cfg(feature = "llvm")]
use crate::llvm::jit_memory_usage;
#[cfg(feature = "llvm")]
use revmc_context::RawEvmCompilerFn;

/// The resident map type: code_hash+spec_id → compiled program.
pub(crate) type ResidentMap = DashMap<RuntimeCacheKey, Arc<CompiledProgram>, DefaultHashBuilder>;

/// Bounded MPMC lock-free lookup-event queue.
///
/// Producers (lookup hot path) push without blocking; on overflow the event is silently dropped
/// (`stats.events_dropped` is bumped). Exact lookup counters are updated before enqueueing.
pub(crate) type EventQueue = ArrayQueue<LookupRequest>;

/// Per-entry metadata tracked alongside the resident map for eviction decisions.
struct ResidentMeta {
    /// When this entry was last hit by a lookup.
    last_hit_at: Instant,
    /// Whether this resident program is AOT or JIT.
    kind: ProgramKind,
    /// Persisted artifact-file length for AOT programs. JIT programs use zero.
    artifact_len: usize,
}

/// Cached persisted-artifact state for one observed key.
#[derive(Clone, Debug)]
enum PersistedAotState {
    /// The hotness threshold has not been reached, so storage has not been touched.
    Unprobed,
    /// Storage was probed and contained no usable artifact. The key may be compiled.
    Absent,
    /// Metadata for an artifact that can be retried without another store access.
    Ready(StoredArtifact),
    /// The artifact can never fit within the configured byte budget.
    Oversized,
}

/// One resident AOT entry considered by the deterministic admission planner.
#[derive(Clone, Copy, Debug)]
struct AotResident {
    key: RuntimeCacheKey,
    last_hit_at: Instant,
    artifact_len: usize,
}

/// Result of checking hard AOT budgets and selecting LRU victims.
#[derive(Debug, PartialEq, Eq)]
enum AotAdmissionPlan {
    Admit(Vec<RuntimeCacheKey>),
    CapacityDeferred,
    Oversized,
}

/// Plans the minimum deterministic AOT eviction set needed for one admission.
///
/// Artifact length is used as a stable budget proxy; it is not exact process RSS.
fn plan_aot_admission(
    tuning: RuntimeTuning,
    resident_aot_entries: usize,
    resident_aot_bytes: usize,
    candidate_len: usize,
    now: Instant,
    mode: AdmitMode,
    mut residents: Vec<AotResident>,
) -> AotAdmissionPlan {
    if tuning.max_resident_aot_bytes > 0 && candidate_len > tuning.max_resident_aot_bytes {
        return AotAdmissionPlan::Oversized;
    }

    let entries_fit = |entries: usize| {
        tuning.max_resident_aot_entries == 0 || entries <= tuning.max_resident_aot_entries
    };
    let bytes_fit =
        |bytes: usize| tuning.max_resident_aot_bytes == 0 || bytes <= tuning.max_resident_aot_bytes;

    let mut projected_entries = resident_aot_entries.saturating_add(1);
    let mut projected_bytes = resident_aot_bytes.saturating_add(candidate_len);
    if entries_fit(projected_entries) && bytes_fit(projected_bytes) {
        return AotAdmissionPlan::Admit(Vec::new());
    }

    if mode == AdmitMode::Observed {
        residents
            .retain(|entry| now.duration_since(entry.last_hit_at) >= tuning.resident_aot_min_idle);
    }
    residents.sort_unstable_by(|a, b| {
        a.last_hit_at
            .cmp(&b.last_hit_at)
            .then_with(|| a.key.code_hash.cmp(&b.key.code_hash))
            .then_with(|| (a.key.spec_id as u8).cmp(&(b.key.spec_id as u8)))
    });

    let mut victims = Vec::new();
    for resident in residents {
        projected_entries = projected_entries.saturating_sub(1);
        projected_bytes = projected_bytes.saturating_sub(resident.artifact_len);
        victims.push(resident.key);
        if entries_fit(projected_entries) && bytes_fit(projected_bytes) {
            return AotAdmissionPlan::Admit(victims);
        }
    }
    AotAdmissionPlan::CapacityDeferred
}

/// No-burst demand-load limiter. Successful acquisitions always schedule from
/// `now`, so idle time never accumulates future capacity.
#[derive(Debug, Default)]
struct AotLoadLimiter {
    next_allowed_at: Option<Instant>,
}

impl AotLoadLimiter {
    fn try_acquire(&mut self, now: Instant, interval: Duration) -> bool {
        if interval.is_zero() {
            return true;
        }
        if self.next_allowed_at.is_some_and(|next| now < next) {
            return false;
        }
        self.next_allowed_at = Some(now + interval);
        true
    }

    fn reset(&mut self) {
        self.next_allowed_at = None;
    }
}

/// Returns the total bytes of JIT-allocated memory via the memory plugin.
fn jit_total_bytes() -> usize {
    #[cfg(feature = "llvm")]
    {
        jit_memory_usage().map(|u| u.total_bytes()).unwrap_or(0)
    }
    #[cfg(not(feature = "llvm"))]
    {
        0
    }
}

#[cfg(feature = "llvm")]
struct JitObjectLinker {
    backend: Option<crate::EvmLlvmBackend>,
}

#[cfg(feature = "llvm")]
impl JitObjectLinker {
    const fn new() -> Self {
        Self { backend: None }
    }

    fn link(
        &mut self,
        success: &JitObjectSuccess,
    ) -> eyre::Result<(EvmCompilerFn, Arc<JitCodeBacking>)> {
        let backend = match &mut self.backend {
            Some(backend) => backend,
            None => self.backend.insert(crate::EvmLlvmBackend::new(false)?),
        };

        let symbol_name = CString::new(success.symbol_name.clone())?;
        let builtin_symbols = success
            .builtin_symbols
            .iter()
            .map(|name| {
                let addr = revmc_builtins::Builtin::parse(name)
                    .ok_or_else(|| eyre::eyre!("unknown builtin symbol: {name}"))?
                    .addr();
                Ok((CString::new(name.as_str())?, addr))
            })
            .collect::<eyre::Result<Vec<_>>>()?;
        let (addr, tracker, jd_guard) = backend.link_jit_object_in_fresh_dylib(
            &symbol_name,
            &success.object_bytes,
            &builtin_symbols,
        )?;
        let func =
            EvmCompilerFn::new(unsafe { std::mem::transmute::<usize, RawEvmCompilerFn>(addr) });
        Ok((func, Arc::new(JitCodeBacking::new(tracker, jd_guard))))
    }
}

#[cfg(not(feature = "llvm"))]
struct JitObjectLinker;

#[cfg(not(feature = "llvm"))]
impl JitObjectLinker {
    const fn new() -> Self {
        Self
    }

    fn link(
        &mut self,
        _success: &JitObjectSuccess,
    ) -> eyre::Result<(EvmCompilerFn, Arc<JitCodeBacking>)> {
        eyre::bail!("LLVM backend not available")
    }
}

/// Commands sent to the backend thread on the bounded command channel.
///
/// Lookup-observed events are NOT carried here — they go through the
/// [`EventQueue`] to avoid waking the backend on every lookup.
pub(crate) enum Command {
    /// Explicit request to JIT-compile a bytecode.
    CompileJit(CompileJitRequest),
    /// Explicit request to prepare AOT artifacts.
    PrepareAot(Vec<PrepareAotRequest>),
    /// Clear the resident compiled map.
    ClearResident,
    /// Clear persisted artifacts from the artifact store.
    ClearPersisted,
    /// Clear both resident and persisted.
    ClearAll,
    /// Pause out-of-process helper execution.
    Pause,
    /// Resume out-of-process helper execution.
    Resume,
    /// Shut down the backend.
    Shutdown,
}

/// An explicit JIT compilation request.
pub(crate) struct CompileJitRequest {
    /// The key to compile for.
    pub(crate) key: RuntimeCacheKey,
    /// The raw bytecode.
    pub(crate) bytecode: Bytes,
    /// Optional notifier for synchronous callers.
    pub(crate) sync_notifier: SyncNotifier,
}

/// An explicit AOT preparation request.
pub(crate) struct PrepareAotRequest {
    /// The key to compile for.
    pub(crate) key: RuntimeCacheKey,
    /// The raw bytecode.
    pub(crate) bytecode: Bytes,
}

/// Per-key state tracked by the backend.
struct EntryState {
    /// Number of observed misses.
    hotness: u32,
    /// Current phase.
    phase: EntryPhase,
    /// The bytecode for this key (captured from a miss event).
    bytecode: Bytes,
    /// When this entry was last observed.
    last_observed_at: Instant,
    /// Sync notifiers waiting for this entry to finish compiling.
    pending_notifiers: Vec<SyncNotifier>,
    /// Cached persisted-AOT probe/admission state.
    persisted_aot: PersistedAotState,
    /// Whether this entry consumes one slot in `max_observed_entries`.
    observed: bool,
    /// Admission semantics to use when an in-flight AOT compilation completes.
    working_mode: AdmitMode,
}

/// Phase of a backend entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EntryPhase {
    /// Not yet hot enough for JIT.
    Cold,
    /// JIT compilation in progress on a worker.
    Working,
}

/// Whether a JIT admission request was triggered by hot-path observation
/// (gated on hotness + cold-entry cap) or by an explicit user request
/// (unconditional, may carry a sync notifier).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AdmitMode {
    Observed,
    Explicit,
}

/// Result of trying to publish one cached persisted AOT artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AotAdmissionOutcome {
    Loaded,
    Deferred,
    Oversized,
    LoadFailed,
}

/// All backend-thread-owned mutable state.
struct BackendState {
    /// Shared state (resident map, event queue, stats).
    inner: Arc<super::BackendShared>,
    /// Per-key metadata for eviction (backend-only).
    resident_meta: HashMap<RuntimeCacheKey, ResidentMeta>,
    /// Per-key tracking state (backend-only).
    entries: HashMap<RuntimeCacheKey, EntryState>,
    /// Number of entries created by observed misses.
    observed_entries: usize,
    /// Number of resident AOT programs.
    resident_aot_entries: usize,
    /// Aggregate artifact-file bytes represented by resident AOT programs.
    resident_aot_bytes: usize,
    /// Demand-load rate limiter.
    aot_load_limiter: AotLoadLimiter,
    /// Worker pool for JIT compilation.
    workers: WorkerPool,
    /// Backend-thread-owned linker for out-of-process JIT objects.
    jit_object_linker: JitObjectLinker,
    /// Receiver for worker results.
    result_rx: chan::Receiver<WorkerResult>,
    /// Artifact store for persisted artifacts.
    store: Option<Arc<dyn ArtifactStore>>,
    /// Tuning knobs.
    tuning: RuntimeTuning,
    /// Whether observed misses compile AOT artifacts instead of JIT code.
    aot: bool,
    /// Number of keys currently in Working phase.
    pending_jobs: usize,
    /// Monotonically increasing generation counter, bumped on clear/invalidation.
    generation: u64,
    /// Last time an eviction sweep was run.
    last_sweep: Instant,
    /// Optional user callback for compilation events.
    on_compilation: Option<Arc<dyn Fn(CompilationEvent) + Send + Sync>>,
    /// Optional user callback for sampled persisted-artifact usage.
    on_artifact_usage: Option<Arc<dyn Fn(ArtifactUsageEvent) + Send + Sync>>,
}

impl BackendState {
    fn handle(&mut self, cmd: Command) -> ControlFlow<()> {
        match cmd {
            Command::CompileJit(req) => self.handle_compile_jit(req),
            Command::PrepareAot(reqs) => self.handle_prepare_aot(reqs),
            Command::ClearResident => self.handle_clear_resident(),
            Command::ClearPersisted => self.handle_clear_persisted(),
            Command::ClearAll => self.handle_clear_all(),
            Command::Pause => self.workers.pause(),
            Command::Resume => self.workers.resume(),
            Command::Shutdown => return ControlFlow::Break(()),
        }
        ControlFlow::Continue(())
    }

    fn tick(&mut self) {
        self.drain_events();
        self.run_eviction_sweep();
        self.update_entry_stats();
    }

    fn update_entry_stats(&self) {
        self.inner.stats.tracked_entries.store(self.entries.len() as u64, Ordering::Relaxed);
        let cold_entries =
            self.entries.values().filter(|entry| entry.phase == EntryPhase::Cold).count();
        self.inner.stats.cold_entries.store(cold_entries as u64, Ordering::Relaxed);
    }

    /// Drains queued misses before sampled resident hits.
    fn drain_events(&mut self) {
        // The separate queues prevent usage bookkeeping from displacing demand-load and
        // compilation signals. Cap each drain so a flood cannot starve commands, worker results,
        // or sweeps; surplus events stay queued for the next iteration.
        for _ in 0..self.tuning.max_events_per_drain {
            let Some(event) = self.inner.miss_events.pop() else { break };
            self.handle_lookup_observed(event);
        }
        for _ in 0..self.tuning.max_events_per_drain {
            let Some(event) = self.inner.hit_events.pop() else { break };
            self.handle_lookup_observed(event);
        }
    }

    fn handle_lookup_observed(&mut self, event: LookupRequest) {
        let hit = event.code.is_empty();
        if hit {
            if let Some(meta) = self.resident_meta.get_mut(&event.key) {
                meta.last_hit_at = Instant::now();
            }
            self.record_artifact_usage(event.key, self.tuning.lookup_hit_sample_rate as u64);
        } else {
            let kind = if self.aot { CompilationKind::Aot } else { CompilationKind::Jit };
            self.try_admit(kind, event.key, event.code, SyncNotifier::none(), AdmitMode::Observed);
        }
    }

    fn record_artifact_usage(&self, key: RuntimeCacheKey, weight: u64) {
        if !self.aot || weight == 0 {
            return;
        }
        let Some(callback) = &self.on_artifact_usage else {
            return;
        };
        callback(ArtifactUsageEvent {
            artifact_key: ArtifactKey {
                runtime: key,
                backend: BackendSelection::Llvm,
                opt_level: self.tuning.aot_opt_level,
            },
            weight,
        });
    }

    fn handle_compile_jit(&mut self, req: CompileJitRequest) {
        let kind = if self.aot { CompilationKind::Aot } else { CompilationKind::Jit };
        self.try_admit(kind, req.key, req.bytecode, req.sync_notifier, AdmitMode::Explicit);
    }

    fn handle_prepare_aot(&mut self, reqs: Vec<PrepareAotRequest>) {
        for req in reqs {
            self.try_admit(
                CompilationKind::Aot,
                req.key,
                req.bytecode,
                SyncNotifier::none(),
                AdmitMode::Explicit,
            );
        }
    }

    /// Common admission path for JIT and AOT compilation requests.
    ///
    /// Handles the cold→working state machine, hotness gating, in-flight
    /// dedup, persisted AOT probing, and worker dispatch. Observed promotion
    /// is gated by hotness; explicit requests are unconditional.
    fn try_admit(
        &mut self,
        kind: CompilationKind,
        key: RuntimeCacheKey,
        bytecode: Bytes,
        sync_notifier: SyncNotifier,
        mode: AdmitMode,
    ) {
        if self.inner.resident.contains_key(&key) {
            sync_notifier.notify();
            return;
        }

        if !self.tuning.should_compile(&bytecode) {
            sync_notifier.notify();
            return;
        }

        let now = Instant::now();
        if mode == AdmitMode::Observed {
            let needs_observed_slot = self.entries.get(&key).is_none_or(|entry| !entry.observed);
            if needs_observed_slot
                && (self.tuning.max_observed_entries == 0
                    || self.observed_entries >= self.tuning.max_observed_entries)
            {
                self.inner.stats.observed_entry_rejections.fetch_add(1, Ordering::Relaxed);
                return;
            }
        }

        let entry = self.entries.entry(key).or_insert_with(|| EntryState {
            hotness: 0,
            phase: EntryPhase::Cold,
            bytecode: bytecode.clone(),
            last_observed_at: now,
            pending_notifiers: Vec::new(),
            persisted_aot: PersistedAotState::Unprobed,
            observed: false,
            working_mode: mode,
        });
        if mode == AdmitMode::Observed && !entry.observed {
            entry.observed = true;
            self.observed_entries += 1;
        }
        entry.last_observed_at = now;

        if entry.phase == EntryPhase::Working {
            if mode == AdmitMode::Explicit {
                entry.working_mode = AdmitMode::Explicit;
            }
            entry.pending_notifiers.push(sync_notifier);
            return;
        }

        if mode == AdmitMode::Observed {
            entry.hotness = entry.hotness.saturating_add(1);
        }

        if kind == CompilationKind::Aot {
            let should_probe = mode == AdmitMode::Explicit
                || (entry.hotness as usize) >= self.tuning.persisted_aot_hot_threshold;
            if matches!(entry.persisted_aot, PersistedAotState::Unprobed) && should_probe {
                let persisted_aot = self.probe_persisted_aot(&key);
                self.entries.get_mut(&key).unwrap().persisted_aot = persisted_aot;
            }

            let persisted_aot = self.entries.get(&key).unwrap().persisted_aot.clone();
            match persisted_aot {
                PersistedAotState::Unprobed => return,
                PersistedAotState::Oversized => {
                    sync_notifier.notify();
                    return;
                }
                PersistedAotState::Ready(stored) => {
                    match self.try_admit_stored_aot(key, &stored, mode, now, true) {
                        AotAdmissionOutcome::Loaded => {
                            self.remove_entry(&key);
                            sync_notifier.notify();
                            return;
                        }
                        AotAdmissionOutcome::Deferred => {
                            sync_notifier.notify();
                            return;
                        }
                        AotAdmissionOutcome::Oversized => {
                            self.entries.get_mut(&key).unwrap().persisted_aot =
                                PersistedAotState::Oversized;
                            sync_notifier.notify();
                            return;
                        }
                        AotAdmissionOutcome::LoadFailed => {
                            // Avoid repeatedly probing/loading the broken file. A subsequent
                            // compilation will overwrite it in the artifact store.
                            self.entries.get_mut(&key).unwrap().persisted_aot =
                                PersistedAotState::Absent;
                        }
                    }
                }
                PersistedAotState::Absent => {}
            }
        }

        let entry = self.entries.get_mut(&key).unwrap();
        if mode == AdmitMode::Observed && (entry.hotness as usize) < self.tuning.jit_hot_threshold {
            return;
        }

        if self.pending_jobs >= self.tuning.jit_max_pending_jobs {
            sync_notifier.notify();
            return;
        }

        let prefix = match kind {
            CompilationKind::Jit => "jit",
            CompilationKind::Aot => "aot",
        };
        let opt_level = match kind {
            CompilationKind::Jit => self.tuning.jit_opt_level,
            CompilationKind::Aot => self.tuning.aot_opt_level,
        };
        let symbol = format!("{prefix}_{:x}_{:?}", key.code_hash, key.spec_id);
        let job = CompileJob {
            kind,
            key,
            bytecode: entry.bytecode.clone(),
            symbol_name: symbol,
            opt_level,
            sync_notifier,
            generation: self.generation,
        };

        match self.workers.try_send(job) {
            Ok(()) => {
                debug!(
                    code_hash = %key.code_hash,
                    spec_id = ?key.spec_id,
                    ?kind,
                    hotness = entry.hotness,
                    pending_jobs = self.pending_jobs + 1,
                    "dispatched compilation",
                );
                entry.phase = EntryPhase::Working;
                entry.working_mode = mode;
                self.pending_jobs += 1;
                self.inner.stats.compilations_dispatched.fetch_add(1, Ordering::Relaxed);
            }
            Err(job) => {
                debug!(code_hash = %key.code_hash, "worker pool saturated, dropping request");
                job.sync_notifier.notify();
            }
        }
    }

    /// Probes persisted AOT storage once and returns cacheable tri-state metadata.
    fn probe_persisted_aot(&self, key: &RuntimeCacheKey) -> PersistedAotState {
        let store = match &self.store {
            Some(store) => store,
            None => return PersistedAotState::Absent,
        };
        let artifact_key = ArtifactKey {
            runtime: *key,
            backend: BackendSelection::Llvm,
            opt_level: self.tuning.aot_opt_level,
        };
        self.inner.stats.persisted_aot_probes.fetch_add(1, Ordering::Relaxed);
        match store.load(&artifact_key) {
            Ok(Some(stored)) => {
                if self.tuning.max_resident_aot_bytes > 0
                    && stored.manifest.artifact_len > self.tuning.max_resident_aot_bytes
                {
                    self.inner.stats.persisted_aot_oversized.fetch_add(1, Ordering::Relaxed);
                    PersistedAotState::Oversized
                } else {
                    PersistedAotState::Ready(stored)
                }
            }
            Ok(None) => {
                self.inner.stats.persisted_aot_probe_misses.fetch_add(1, Ordering::Relaxed);
                PersistedAotState::Absent
            }
            Err(error) => {
                warn!(
                    code_hash = %key.code_hash,
                    error = %error,
                    "failed to probe artifact store; recompilation remains available",
                );
                PersistedAotState::Absent
            }
        }
    }

    /// Attempts to load and publish cached persisted AOT metadata under the
    /// configured hard budgets and demand-admission policy.
    fn try_admit_stored_aot(
        &mut self,
        key: RuntimeCacheKey,
        stored: &StoredArtifact,
        mode: AdmitMode,
        now: Instant,
        persisted_candidate: bool,
    ) -> AotAdmissionOutcome {
        let residents = self
            .resident_meta
            .iter()
            .filter(|(_, meta)| meta.kind == ProgramKind::Aot)
            .map(|(key, meta)| AotResident {
                key: *key,
                last_hit_at: meta.last_hit_at,
                artifact_len: meta.artifact_len,
            })
            .collect();
        let victims = match plan_aot_admission(
            self.tuning,
            self.resident_aot_entries,
            self.resident_aot_bytes,
            stored.manifest.artifact_len,
            now,
            mode,
            residents,
        ) {
            AotAdmissionPlan::Admit(victims) => victims,
            AotAdmissionPlan::CapacityDeferred => {
                self.inner.stats.persisted_aot_capacity_deferred.fetch_add(1, Ordering::Relaxed);
                return AotAdmissionOutcome::Deferred;
            }
            AotAdmissionPlan::Oversized => {
                self.inner.stats.persisted_aot_oversized.fetch_add(1, Ordering::Relaxed);
                return AotAdmissionOutcome::Oversized;
            }
        };

        if persisted_candidate
            && mode == AdmitMode::Observed
            && !self.aot_load_limiter.try_acquire(now, self.tuning.persisted_aot_load_interval)
        {
            self.inner.stats.persisted_aot_rate_limited.fetch_add(1, Ordering::Relaxed);
            return AotAdmissionOutcome::Deferred;
        }

        let load_started = Instant::now();
        let program = match Self::load_aot_program(key, stored) {
            Ok(program) => program,
            Err(error) => {
                warn!(
                    code_hash = %key.code_hash,
                    error = %error,
                    "failed to load persisted AOT artifact; recompilation remains available",
                );
                return AotAdmissionOutcome::LoadFailed;
            }
        };
        let load_ns = load_started.elapsed().as_nanos().min(u64::MAX as u128) as u64;

        // Dlopen happens before eviction so a load failure cannot create a cache hole.
        for victim in victims {
            self.remove_resident(&victim);
            self.remove_entry(&victim);
            self.inner.stats.aot_evictions.fetch_add(1, Ordering::Relaxed);
            self.inner.stats.evictions.fetch_add(1, Ordering::Relaxed);
        }
        self.insert_resident(key, Arc::new(program));
        if persisted_candidate {
            self.record_artifact_usage(key, 1);
        }
        if persisted_candidate && mode == AdmitMode::Observed {
            self.inner.stats.persisted_aot_loads.fetch_add(1, Ordering::Relaxed);
            self.inner.stats.persisted_aot_load_ns.fetch_add(load_ns, Ordering::Relaxed);
        }
        AotAdmissionOutcome::Loaded
    }

    fn load_aot_program(
        key: RuntimeCacheKey,
        stored: &StoredArtifact,
    ) -> eyre::Result<CompiledProgram> {
        let library = unsafe { libloading::Library::new(&stored.dylib_path) }
            .map_err(|error| eyre::eyre!("dlopen {:?}: {error}", stored.dylib_path))?;
        let func: EvmCompilerFn = unsafe {
            let symbol: libloading::Symbol<'_, EvmCompilerFn> =
                library.get(stored.manifest.symbol_name.as_bytes()).map_err(|error| {
                    eyre::eyre!("symbol '{}': {error}", stored.manifest.symbol_name)
                })?;
            *symbol
        };
        Ok(CompiledProgram::new_aot(
            key,
            func,
            Arc::new(LoadedLibrary::new(library)),
            stored.manifest.artifact_len,
        ))
    }

    fn handle_clear_resident(&mut self) {
        self.workers.cancel_in_flight();
        self.inner.resident.clear();
        self.resident_meta.clear();
        self.resident_aot_entries = 0;
        self.resident_aot_bytes = 0;
        self.aot_load_limiter.reset();
        // Notify any pending sync callers before clearing entries.
        for (_, entry) in self.entries.drain() {
            for n in entry.pending_notifiers {
                n.notify();
            }
        }
        self.observed_entries = 0;
        // Discard pending lookup events: they were observed before the clear
        // and would otherwise get processed against the new generation.
        while self.inner.miss_events.pop().is_some() {}
        while self.inner.hit_events.pop().is_some() {}
        // Bump generation so in-flight worker results from before the clear are discarded.
        self.generation += 1;
        debug!(generation = self.generation, "resident map cleared");
    }

    fn handle_clear_persisted(&mut self) {
        if let Some(store) = &self.store {
            if let Err(e) = store.clear() {
                warn!(error = %e, "failed to clear artifact store");
            } else {
                debug!("artifact store cleared");
            }
        }
    }

    fn handle_clear_all(&mut self) {
        self.handle_clear_resident();
        self.handle_clear_persisted();
    }

    fn insert_resident(&mut self, key: RuntimeCacheKey, program: Arc<CompiledProgram>) {
        if self.resident_meta.contains_key(&key) {
            self.remove_resident(&key);
        }
        let kind = program.kind;
        let artifact_len = program.artifact_len;
        if kind == ProgramKind::Aot {
            self.resident_aot_entries = self.resident_aot_entries.saturating_add(1);
            self.resident_aot_bytes = self.resident_aot_bytes.saturating_add(artifact_len);
        }
        self.inner.resident.insert(key, program);
        self.resident_meta
            .insert(key, ResidentMeta { last_hit_at: Instant::now(), kind, artifact_len });
    }

    fn remove_resident(&mut self, key: &RuntimeCacheKey) {
        self.inner.resident.remove(key);
        if let Some(meta) = self.resident_meta.remove(key)
            && meta.kind == ProgramKind::Aot
        {
            self.resident_aot_entries = self.resident_aot_entries.saturating_sub(1);
            self.resident_aot_bytes = self.resident_aot_bytes.saturating_sub(meta.artifact_len);
        }
    }

    fn remove_entry(&mut self, key: &RuntimeCacheKey) -> Option<EntryState> {
        let entry = self.entries.remove(key)?;
        if entry.observed {
            self.observed_entries = self.observed_entries.saturating_sub(1);
        }
        Some(entry)
    }

    fn handle_worker_result(&mut self, result: WorkerResult) {
        self.pending_jobs = self.pending_jobs.saturating_sub(1);

        // Drain pending notifiers from the entry before processing.
        let pending_notifiers = self
            .entries
            .get_mut(&result.key)
            .map(|e| mem::take(&mut e.pending_notifiers))
            .unwrap_or_default();

        let notify = || {
            result.sync_notifier.notify();
            for n in pending_notifiers {
                n.notify();
            }
        };

        // Discard stale results from a previous generation (e.g. after clear).
        if result.generation != self.generation {
            debug!(
                code_hash = %result.key.code_hash,
                result_gen = result.generation,
                current_gen = self.generation,
                "discarding stale worker result",
            );
            self.remove_entry(&result.key);
            notify();
            return;
        }

        let kind = result.kind;
        let success = result.outcome.is_ok();

        if let Some(cb) = &self.on_compilation {
            cb(CompilationEvent {
                code_hash: result.key.code_hash,
                spec_id: result.key.spec_id,
                duration: result.compile_duration,
                kind,
                success,
                timings: result.timings,
            });
        }

        match result.outcome {
            Ok(WorkerSuccess::Jit(success)) => {
                let program =
                    Arc::new(CompiledProgram::new_jit(result.key, success.func, success.backing));
                self.insert_resident(result.key, program);
                self.remove_entry(&result.key);
                self.inner.stats.compilations_succeeded.fetch_add(1, Ordering::Relaxed);

                debug!(
                    code_hash = %result.key.code_hash,
                    spec_id = ?result.key.spec_id,
                    compile_time = ?result.compile_duration,
                    "JIT program published to resident map",
                );
            }
            Ok(WorkerSuccess::Aot(success)) => {
                self.handle_aot_success(result.key, success);
            }
            Ok(WorkerSuccess::JitObject(success)) => {
                self.handle_jit_object_success(result.key, success, result.compile_duration);
            }
            Err(err) => {
                self.remove_entry(&result.key);
                self.inner.stats.compilations_failed.fetch_add(1, Ordering::Relaxed);

                warn!(
                    code_hash = %result.key.code_hash,
                    error = %err,
                    compile_time = ?result.compile_duration,
                    "compilation failed",
                );
            }
        }

        notify();
    }

    fn handle_jit_object_success(
        &mut self,
        key: RuntimeCacheKey,
        success: JitObjectSuccess,
        compile_duration: std::time::Duration,
    ) {
        match self.jit_object_linker.link(&success) {
            Ok((func, backing)) => {
                let program = Arc::new(CompiledProgram::new_jit(key, func, backing));
                self.insert_resident(key, program);
                self.remove_entry(&key);
                self.inner.stats.compilations_succeeded.fetch_add(1, Ordering::Relaxed);

                debug!(
                    code_hash = %key.code_hash,
                    spec_id = ?key.spec_id,
                    compile_time = ?compile_duration,
                    object_len = success.object_bytes.len(),
                    "JIT object linked and published to resident map",
                );
            }
            Err(err) => {
                self.remove_entry(&key);
                self.inner.stats.compilations_failed.fetch_add(1, Ordering::Relaxed);

                warn!(
                    code_hash = %key.code_hash,
                    error = %err,
                    compile_time = ?compile_duration,
                    "failed to link JIT object",
                );
            }
        }
    }

    fn handle_aot_success(&mut self, key: RuntimeCacheKey, success: AotSuccess) {
        let mode =
            self.entries.get(&key).map(|entry| entry.working_mode).unwrap_or(AdmitMode::Explicit);
        let artifact_key = ArtifactKey {
            runtime: key,
            backend: BackendSelection::Llvm,
            opt_level: self.tuning.aot_opt_level,
        };

        let content_hash = keccak256(&success.dylib_bytes).0;

        let manifest = ArtifactManifest {
            artifact_key: artifact_key.clone(),
            symbol_name: success.symbol_name.clone(),
            bytecode_len: success.bytecode_len,
            artifact_len: success.dylib_bytes.len(),
            created_at_unix_secs: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            content_hash,
        };

        let Some(store) = self.store.clone() else {
            warn!(
                code_hash = %key.code_hash,
                "AOT compilation completed but no artifact store configured",
            );
            self.remove_entry(&key);
            self.inner.stats.compilations_failed.fetch_add(1, Ordering::Relaxed);
            return;
        };

        // Persist before applying resident budgets. A deferred artifact remains reusable on disk.
        if let Err(error) = store.store(&artifact_key, &manifest, &success.dylib_bytes) {
            warn!(
                code_hash = %key.code_hash,
                error = %error,
                "failed to persist AOT artifact",
            );
            self.remove_entry(&key);
            self.inner.stats.compilations_failed.fetch_add(1, Ordering::Relaxed);
            return;
        }
        debug!(
            code_hash = %key.code_hash,
            spec_id = ?key.spec_id,
            dylib_len = success.dylib_bytes.len(),
            "AOT artifact persisted to store",
        );

        let stored = match store.load(&artifact_key) {
            Ok(Some(stored)) => stored,
            Ok(None) => {
                warn!(code_hash = %key.code_hash, "stored AOT artifact not found on reload");
                self.remove_entry(&key);
                self.inner.stats.compilations_failed.fetch_add(1, Ordering::Relaxed);
                return;
            }
            Err(error) => {
                warn!(
                    code_hash = %key.code_hash,
                    error = %error,
                    "failed to reload persisted AOT artifact",
                );
                self.remove_entry(&key);
                self.inner.stats.compilations_failed.fetch_add(1, Ordering::Relaxed);
                return;
            }
        };

        if let Some(entry) = self.entries.get_mut(&key) {
            entry.phase = EntryPhase::Cold;
            entry.persisted_aot = PersistedAotState::Ready(stored.clone());
        }

        match self.try_admit_stored_aot(key, &stored, mode, Instant::now(), false) {
            AotAdmissionOutcome::Loaded => {
                self.remove_entry(&key);
                self.inner.stats.compilations_succeeded.fetch_add(1, Ordering::Relaxed);
                debug!(
                    code_hash = %key.code_hash,
                    spec_id = ?key.spec_id,
                    "AOT program loaded into resident map",
                );
            }
            AotAdmissionOutcome::Deferred => {
                // The ready metadata remains cached and will be retried on a later miss.
                self.inner.stats.compilations_succeeded.fetch_add(1, Ordering::Relaxed);
            }
            AotAdmissionOutcome::Oversized => {
                if let Some(entry) = self.entries.get_mut(&key) {
                    entry.persisted_aot = PersistedAotState::Oversized;
                }
                self.inner.stats.compilations_succeeded.fetch_add(1, Ordering::Relaxed);
            }
            AotAdmissionOutcome::LoadFailed => {
                if let Some(entry) = self.entries.get_mut(&key) {
                    entry.persisted_aot = PersistedAotState::Absent;
                }
                self.inner.stats.compilations_failed.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Runs an eviction sweep: removes idle entries and enforces the memory budget.
    fn run_eviction_sweep(&mut self) {
        if !self.should_sweep() {
            return;
        }

        let now = Instant::now();
        self.last_sweep = now;

        let idle_duration = self.tuning.idle_evict_duration;
        let cold_idle_duration = self.tuning.cold_entry_idle_duration;
        let budget = self.tuning.resident_code_cache_bytes;

        // Phase 1: evict idle entries.
        if let Some(idle) = idle_duration {
            let idle_keys: Vec<RuntimeCacheKey> = self
                .resident_meta
                .iter()
                .filter(|(_, meta)| now.duration_since(meta.last_hit_at) > idle)
                .map(|(key, _)| *key)
                .collect();

            for key in &idle_keys {
                debug!(
                    code_hash = %key.code_hash,
                    spec_id = ?key.spec_id,
                    "evicting idle entry",
                );
                self.remove_resident(key);
                self.remove_entry(key);
                self.inner.stats.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }

        // Phase 2: evict stale cold entries that never became hot enough to compile.
        if let Some(idle) = cold_idle_duration {
            let stale_keys = self
                .entries
                .iter()
                .filter(|(key, entry)| {
                    entry.phase == EntryPhase::Cold
                        && now.duration_since(entry.last_observed_at) > idle
                        && !matches!(entry.persisted_aot, PersistedAotState::Oversized)
                        && !self.inner.resident.contains_key(key)
                })
                .map(|(key, _)| *key)
                .collect::<Vec<_>>();
            for key in &stale_keys {
                self.remove_entry(key);
            }
            self.inner
                .stats
                .cold_entry_evictions
                .fetch_add(stale_keys.len() as u64, Ordering::Relaxed);
        }

        // Phase 3: enforce memory budget by evicting LRU JIT entries.
        if budget > 0 && jit_total_bytes() > budget {
            // Collect JIT entries sorted by last_hit_at ascending (oldest first).
            // AOT entries are excluded because they don't contribute to `jit_total_bytes()`.
            let mut entries: Vec<(RuntimeCacheKey, Instant)> = self
                .resident_meta
                .iter()
                .filter(|(_, meta)| meta.kind == ProgramKind::Jit)
                .map(|(key, meta)| (*key, meta.last_hit_at))
                .collect();
            entries.sort_by_key(|(_, t)| *t);

            for (key, _) in entries {
                if jit_total_bytes() <= budget {
                    break;
                }
                debug!(
                    code_hash = %key.code_hash,
                    spec_id = ?key.spec_id,
                    "evicting entry to stay within memory budget",
                );
                self.remove_resident(&key);
                self.remove_entry(&key);
                self.inner.stats.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Returns whether eviction is configured and a sweep is due.
    fn should_sweep(&self) -> bool {
        // Over budget — sweep immediately regardless of interval.
        let maxrss = self.tuning.resident_code_cache_bytes;
        if maxrss > 0 && jit_total_bytes() > maxrss {
            return true;
        }
        (self.tuning.idle_evict_duration.is_some()
            || self.tuning.cold_entry_idle_duration.is_some())
            && self.last_sweep.elapsed() >= self.tuning.eviction_sweep_interval
    }
}

/// Runs the backend event loop. Called on the backend thread.
pub(crate) fn run(
    inner: Arc<super::BackendShared>,
    cmd_rx: chan::Receiver<Command>,
    config: RuntimeConfig,
) {
    debug!("backend thread started");

    let (result_tx, result_rx) = chan::unbounded::<WorkerResult>();

    let workers = WorkerPool::new(result_tx, config.clone(), Arc::clone(&inner.stats));

    let sweep_interval = config.tuning.eviction_sweep_interval;
    let event_drain_interval = config.tuning.event_drain_interval;

    // Seed resident metadata from startup-preloaded AOT entries.
    let now = Instant::now();
    let mut preload_meta = HashMap::default();
    let mut resident_aot_entries = 0usize;
    let mut resident_aot_bytes = 0usize;
    for entry in inner.resident.iter() {
        let kind = entry.kind;
        let artifact_len = entry.artifact_len;
        if kind == ProgramKind::Aot {
            resident_aot_entries = resident_aot_entries.saturating_add(1);
            resident_aot_bytes = resident_aot_bytes.saturating_add(artifact_len);
        }
        preload_meta.insert(*entry.key(), ResidentMeta { last_hit_at: now, kind, artifact_len });
    }

    let mut state = BackendState {
        inner,
        resident_meta: preload_meta,
        entries: HashMap::default(),
        observed_entries: 0,
        resident_aot_entries,
        resident_aot_bytes,
        aot_load_limiter: AotLoadLimiter::default(),
        workers,
        jit_object_linker: JitObjectLinker::new(),
        result_rx,
        store: config.store,
        tuning: config.tuning,
        aot: config.aot,
        pending_jobs: 0,
        generation: 0,
        last_sweep: now,
        on_compilation: config.on_compilation,
        on_artifact_usage: config.on_artifact_usage,
    };

    // Tick interval is min(event_drain, sweep) so we never sleep longer than
    // either. Events are drained on every wakeup regardless of cause.
    let tick = event_drain_interval.min(sweep_interval);
    let shutdown_reason;

    loop {
        chan::select! {
            recv(cmd_rx) -> msg => {
                let Ok(cmd) = msg else {
                    shutdown_reason = "channel closed";
                    break;
                };
                if state.handle(cmd).is_break() {
                    shutdown_reason = "shutdown command";
                    break;
                }
            }
            recv(state.result_rx) -> msg => {
                match msg {
                    Ok(result) => state.handle_worker_result(result),
                    Err(_) => warn!("worker unexpectedly closed"),
                }
            }
            default(tick) => {}
        }
        state.tick();
    }

    debug!(?shutdown_reason, stats = ?state.inner.stats(), "backend task shutting down");

    state.workers.shutdown();
    while state.result_rx.try_recv().is_ok() {}
}

#[cfg(all(test, feature = "llvm"))]
mod tests {
    use super::*;
    use crate::runtime::worker::{
        CompileJob, WorkerSuccess, compile_jit_object_artifact, create_compiler,
    };
    use revm_primitives::hardfork::SpecId;

    /// PUSH1 0x42 PUSH0 MSTORE PUSH1 0x20 PUSH0 RETURN.
    const BYTECODE_RET42: &[u8] = &[0x60, 0x42, 0x5f, 0x52, 0x60, 0x20, 0x5f, 0xf3];

    fn compile_jit_object(symbol_name: &str) -> JitObjectSuccess {
        let config = RuntimeConfig::default();
        let mut compiler = create_compiler(&config, true).unwrap();
        let job = CompileJob {
            kind: CompilationKind::Jit,
            key: RuntimeCacheKey { code_hash: keccak256(BYTECODE_RET42), spec_id: SpecId::CANCUN },
            bytecode: Bytes::copy_from_slice(BYTECODE_RET42),
            symbol_name: symbol_name.to_owned(),
            opt_level: config.tuning.jit_opt_level,
            sync_notifier: SyncNotifier::none(),
            generation: 0,
        };

        match compile_jit_object_artifact(&job, &mut compiler).unwrap() {
            WorkerSuccess::JitObject(success) => success,
            _ => unreachable!(),
        }
    }

    #[test]
    fn jit_object_linker_relinks_live_symbol_name() {
        let success = compile_jit_object("jit_duplicate_symbol");
        let success2 = JitObjectSuccess {
            symbol_name: success.symbol_name.clone(),
            object_bytes: success.object_bytes.clone(),
            builtin_symbols: success.builtin_symbols.clone(),
        };

        let mut linker = JitObjectLinker::new();
        let (_first_func, _first_backing) = linker.link(&success).unwrap();
        let (_second_func, _second_backing) = linker
            .link(&success2)
            .expect("same symbol should link while previous backing remains alive");
    }
}

#[cfg(test)]
mod admission_policy_tests {
    use super::*;
    use alloy_primitives::B256;
    use revm_primitives::hardfork::SpecId;

    fn key(last_byte: u8) -> RuntimeCacheKey {
        RuntimeCacheKey { code_hash: B256::with_last_byte(last_byte), spec_id: SpecId::CANCUN }
    }

    fn resident(key: RuntimeCacheKey, last_hit_at: Instant, artifact_len: usize) -> AotResident {
        AotResident { key, last_hit_at, artifact_len }
    }

    #[test]
    fn entry_and_byte_budgets_are_independent() {
        let now = Instant::now();
        let by_entries = RuntimeTuning { max_resident_aot_entries: 2, ..Default::default() };
        assert_eq!(
            plan_aot_admission(
                by_entries,
                2,
                20,
                10,
                now,
                AdmitMode::Explicit,
                vec![resident(key(1), now, 10), resident(key(2), now, 10)],
            ),
            AotAdmissionPlan::Admit(vec![key(1)]),
        );

        let by_bytes = RuntimeTuning { max_resident_aot_bytes: 100, ..Default::default() };
        assert_eq!(
            plan_aot_admission(
                by_bytes,
                2,
                100,
                70,
                now,
                AdmitMode::Explicit,
                vec![resident(key(1), now, 40), resident(key(2), now, 60)],
            ),
            AotAdmissionPlan::Admit(vec![key(1), key(2)]),
        );
    }

    #[test]
    fn demand_respects_idle_age_while_explicit_bypasses_it() {
        let last_hit_at = Instant::now();
        let almost_idle = last_hit_at + Duration::from_secs(299);
        let tuning = RuntimeTuning {
            max_resident_aot_entries: 1,
            resident_aot_min_idle: Duration::from_secs(300),
            ..Default::default()
        };
        let residents = vec![resident(key(1), last_hit_at, 10)];

        assert_eq!(
            plan_aot_admission(
                tuning,
                1,
                10,
                10,
                almost_idle,
                AdmitMode::Observed,
                residents.clone(),
            ),
            AotAdmissionPlan::CapacityDeferred,
        );
        assert_eq!(
            plan_aot_admission(tuning, 1, 10, 10, almost_idle, AdmitMode::Explicit, residents,),
            AotAdmissionPlan::Admit(vec![key(1)]),
        );
    }

    #[test]
    fn oversized_candidate_is_rejected_without_victims() {
        let tuning = RuntimeTuning { max_resident_aot_bytes: 100, ..Default::default() };
        assert_eq!(
            plan_aot_admission(tuning, 0, 0, 101, Instant::now(), AdmitMode::Explicit, Vec::new(),),
            AotAdmissionPlan::Oversized,
        );
    }

    #[test]
    fn rate_limiter_allows_one_per_interval_without_burst() {
        let start = Instant::now();
        let interval = Duration::from_millis(100);
        let mut limiter = AotLoadLimiter::default();

        assert!(limiter.try_acquire(start, interval));
        assert!(!limiter.try_acquire(start + Duration::from_millis(99), interval));
        assert!(limiter.try_acquire(start + interval, interval));

        let after_idle = start + Duration::from_secs(10);
        assert!(limiter.try_acquire(after_idle, interval));
        assert!(!limiter.try_acquire(after_idle, interval));
    }
}
