//! gigatoken-backed [`TextTokenizer`] for the encode pool.
//!
//! On the Rust tokenizer-manager, `tokenize` is 93-97% of everything this side
//! spends at prompts of 16K tokens and up (0.53ms at in=256 rising to 492ms at
//! in=256K, against 0.015ms of detokenize and 0.07ms of msgpack encode), so the
//! encoder is the only stage here worth replacing. gigatoken encodes a single
//! document ~100x faster per token than `dynamo-tokenizers` does.
//!
//! Three properties of gigatoken shape this wrapper:
//!
//! * Its encoder mutates a per-instance pretoken cache (`&mut self`), while
//!   [`TextTokenizer`] hands one `Send + Sync` instance to every pinned worker.
//!   Each worker thread therefore gets its own `fork()` — forks share the
//!   immutable vocab/merge tables and keep independent caches, which is how
//!   gigatoken is meant to be used across threads anyway.
//! * Its `encode` applies no post-processor, i.e. it is HuggingFace's
//!   `add_special_tokens=False`. The BOS/EOS the dynamo backend adds is
//!   reproduced from the affix pair [`super::derive_affixes`] derives, which
//!   also verifies the ids match the dynamo backend token for token.
//! * Only byte-level BPE vocabularies are accepted. gigatoken's SentencePiece
//!   backend is both its least optimized path and the one whose decode diverges
//!   from HuggingFace, and it is the only part of the crate needing nightly
//!   Rust, so the encode-side win does not justify carrying it.
//!
//! Detokenization deliberately stays on `dynamo-tokenizers`: gigatoken exposes
//! no incremental decoder (only whole-sequence `decode`), the streaming path
//! needs the partial-UTF-8 buffering `DecodeStream` already does, and decode is
//! a rounding error in the measured budget.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

// The crate-root re-export: gigatoken's `bpe` module is `pub(crate)`.
use gigatoken_rs::Tokenizer as GtTokenizer;
use gigatoken_rs::load_tokenizer::hf::{HfTokenizer, load_hf_slice};

use crate::error::Error;
use crate::message::TokenIds;
use crate::tokenizer::{TextTokenizer, derive_affixes};

/// Pretoken-cache budget per worker thread. gigatoken's own default is 512 MiB
/// per encode worker, sized for offline corpus passes; a server multiplies that
/// by `tokenizer_worker_num` and holds it for the process lifetime, so cap it
/// far lower. Serving hit rates are driven by the head of the pretoken
/// distribution, which fits easily; a wipe only costs recomputing previously
/// cached pretokens and never changes output.
const CACHE_BYTES_PER_WORKER: usize = 32 << 20;

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// Per-thread encode state, keyed by owning [`GigatokenTokenizer`]. Only the
    /// pinned pool and the MM worker path encode, so this stays at one entry in
    /// practice; keying keeps it correct if a second instance ever exists.
    static FORKS: RefCell<HashMap<u64, GtTokenizer>> = RefCell::new(HashMap::new());
}

pub struct GigatokenTokenizer {
    /// Forked per worker thread; never encoded through directly.
    template: GtTokenizer,
    /// Identifies this instance's entry in [`FORKS`].
    id: u64,
    /// Ids the reference tokenizer's post-processor puts before/after a
    /// sequence, which gigatoken does not add itself.
    prefix: Vec<i32>,
    suffix: Vec<i32>,
}

impl GigatokenTokenizer {
    /// Load from `tokenizer.json` bytes and verify against `reference`.
    ///
    /// `Err` means "do not use gigatoken for this model" — an unsupported
    /// vocabulary, or ids that do not match the reference. The caller logs it
    /// and keeps the reference tokenizer, so the flag costs speed, never
    /// correctness.
    pub fn load(tokenizer_json: &[u8], reference: &dyn TextTokenizer) -> Result<Self, String> {
        let mut template = match load_hf_slice(tokenizer_json)
            .map_err(|e| format!("gigatoken could not load tokenizer.json: {e}"))?
        {
            HfTokenizer::Bpe(t) => t,
            HfTokenizer::SentencePiece(_) => {
                return Err("gigatoken: SentencePiece vocabulary is not supported here".to_string());
            }
        };
        template.set_max_cache_bytes(Some(CACHE_BYTES_PER_WORKER));

        // Borrowed mutably by the probe encoder, so derive before moving in.
        let (prefix, suffix) = {
            let probe = &mut template;
            derive_affixes(reference, |text| encode_bare(probe, text))
                .map_err(|e| format!("gigatoken: {e}"))?
        };
        Ok(Self {
            template,
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            prefix,
            suffix,
        })
    }

    /// Run `f` against this thread's fork, creating it on first use.
    fn with_fork<R>(&self, f: impl FnOnce(&mut GtTokenizer) -> R) -> R {
        FORKS.with(|forks| {
            let mut forks = forks.borrow_mut();
            let tok = forks.entry(self.id).or_insert_with(|| self.template.fork());
            f(tok)
        })
    }

    /// The affixes this backend adds, for startup logging.
    pub fn affixes(&self) -> (&[i32], &[i32]) {
        (&self.prefix, &self.suffix)
    }
}

impl TextTokenizer for GigatokenTokenizer {
    fn encode(&self, text: &str) -> Result<TokenIds, Error> {
        if text.is_empty() {
            // Same 400-not-500 behavior as the dynamo backend.
            return Err(Error::Validation("prompt cannot be empty".into()));
        }
        let raw = self.with_fork(|tok| encode_bare(tok, text));
        let mut ids = Vec::with_capacity(self.prefix.len() + raw.len() + self.suffix.len());
        ids.extend_from_slice(&self.prefix);
        ids.extend_from_slice(&raw);
        ids.extend_from_slice(&self.suffix);
        Ok(ids)
    }

    /// The derived prefix, matching what the reference tokenizer prepends —
    /// `TokenizerWorker` strips it for template-rendered prompts.
    fn auto_specials(&self) -> Vec<i32> {
        self.prefix.clone()
    }
}

/// Bare ids for `text`, with no affixes — the `add_special_tokens=False` form.
/// Vocab ids are non-negative and fit in i32.
fn encode_bare(tok: &mut GtTokenizer, text: &str) -> Vec<i32> {
    let mut raw = Vec::new();
    tok.encode_with_added_tokens_flat(text.as_bytes(), &mut raw);
    raw.into_iter().map(|id| id as i32).collect()
}
