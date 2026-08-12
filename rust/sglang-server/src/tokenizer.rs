//! Tokenizer pool — CPU-bound, runs on pinned OS threads (off the async
//! executor). Each worker pulls a `Request` from the shared `flume` receiver,
//! fills `input_ids`, and moves the request back to the TokenizerManager inbox.
//!
//! The text→ids step is behind [`TextTokenizer`], implemented by
//! [`DynamoTokenizer`] (dynamo-tokenizers: HuggingFace / tiktoken / fastokens).
//! A non-skip server requires a real tokenizer (enforced at startup); under
//! `skip_tokenizer_init` the pool isn't spawned at all.
//!
//! Mirrors the Python `_tokenize_one_request` text path: when the request
//! already carries `input_ids` it skips tokenization (handled upstream in the
//! TokenizerManager `classify`); otherwise the prompt text is encoded here.

use std::path::Path;
use std::sync::Arc;

use crate::error::Error;
use crate::fsm::Event;
use crate::message::{Request, RequestKind, TokenIds};
use crate::runtime::Runnable;
use crate::tokenizer_manager::TmEvent;

#[cfg(feature = "gigatoken")]
pub mod gigatoken;

/// Pluggable text→token-ids backend. `Send + Sync` so one instance is shared
/// (read-only) across all pinned workers.
pub trait TextTokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Result<TokenIds, Error>;

    /// The special tokens this tokenizer auto-prepends on every `encode` —
    /// Python's `encode("")` probe (`serving_chat._tokenizer_auto_adds_specials`).
    /// Empty when it adds none (tiktoken backends, no BOS/EOS post-processor).
    fn auto_specials(&self) -> Vec<i32> {
        Vec::new()
    }
}

/// Load the tokenizer shared (Arc-backed) by the encode pool and detok shards.
/// `None` under `skip_tokenizer_init`, else required (missing/failed load → `Err`).
/// `tokenizer_path` is a tokenizer file, a model dir, or an HF Hub repo id
/// (resolved from the local cache — no network).
pub fn load_tokenizer(
    tokenizer_path: Option<&str>,
    revision: Option<&str>,
    skip_tokenizer_init: bool,
) -> Result<Option<dynamo_tokenizers::Tokenizer>, String> {
    if skip_tokenizer_init {
        tracing::info!("skip_tokenizer_init: token ids in and out; no tokenizer/detokenizer");
        return Ok(None);
    }
    let path = tokenizer_path.ok_or_else(|| {
        "no tokenizer configured: set tokenizer_path or enable skip_tokenizer_init".to_string()
    })?;
    let file = resolve_model_file(path, revision, "tokenizer.json")
        .ok_or_else(|| format!("tokenizer.json not found for '{path}'"))?;
    let tokenizer = dynamo_tokenizers::Tokenizer::from_file_with_options(
        &file,
        dynamo_tokenizers::TokenizerOptions {
            add_special_tokens: true,
        },
    )
    .map_err(|e| format!("tokenizer load failed ({file}): {e}"))?;
    tracing::info!(%path, "loaded tokenizer");
    Ok(Some(tokenizer))
}

/// The `TextTokenizer` the encode pool runs on, selected by `--tokenizer-backend`.
///
/// `huggingface` (the default) wraps the already-loaded dynamo tokenizer.
/// `gigatoken` replaces only the encode step — the stage that dominates this
/// side's latency — and verifies its ids against the dynamo tokenizer at load.
/// Anything gigatoken cannot back byte-identically (unsupported vocabulary, ids
/// that do not match, or a build without the `gigatoken` cargo feature) logs why
/// and keeps the dynamo tokenizer, so the flag costs speed and never correctness.
pub fn build_text_tokenizer(
    loaded: &dynamo_tokenizers::Tokenizer,
    tokenizer_path: &str,
    revision: Option<&str>,
    tokenizer_backend: &str,
) -> Arc<dyn TextTokenizer> {
    let reference = Arc::new(DynamoTokenizer::new(loaded.clone()));
    if tokenizer_backend != "gigatoken" {
        return reference;
    }
    #[cfg(not(feature = "gigatoken"))]
    {
        let _ = (tokenizer_path, revision);
        tracing::warn!(
            "--tokenizer-backend=gigatoken was requested but this build has no \
             gigatoken support (cargo feature 'gigatoken' off); using the default \
             tokenizer"
        );
        reference
    }
    #[cfg(feature = "gigatoken")]
    {
        match load_gigatoken(tokenizer_path, revision, reference.as_ref()) {
            Ok(tokenizer) => {
                let (prefix, suffix) = tokenizer.affixes();
                tracing::info!(
                    ?prefix,
                    ?suffix,
                    "gigatoken encode backend enabled (ids verified against the \
                     default tokenizer); detokenization stays on the default backend"
                );
                Arc::new(tokenizer)
            }
            Err(why) => {
                tracing::warn!("{why}; using the default tokenizer for encode");
                reference
            }
        }
    }
}

#[cfg(feature = "gigatoken")]
fn load_gigatoken(
    tokenizer_path: &str,
    revision: Option<&str>,
    reference: &dyn TextTokenizer,
) -> Result<gigatoken::GigatokenTokenizer, String> {
    let file = resolve_model_file(tokenizer_path, revision, "tokenizer.json")
        .ok_or_else(|| format!("gigatoken: tokenizer.json not found for '{tokenizer_path}'"))?;
    let bytes = std::fs::read(&file).map_err(|e| format!("gigatoken: reading {file}: {e}"))?;
    gigatoken::GigatokenTokenizer::load(&bytes, reference)
}

/// Resolve a model file from the tokenizer source: a dir → `dir/<file>`, a file →
/// its sibling, else an HF Hub repo id → the local cache. `None` if not found.
pub fn resolve_model_file(path: &str, revision: Option<&str>, filename: &str) -> Option<String> {
    let p = Path::new(path);
    if p.is_dir() {
        let f = p.join(filename);
        return f.is_file().then(|| f.to_string_lossy().into_owned());
    }
    if p.is_file() {
        // `path` is a file (e.g. `tokenizer.json`); look for the sibling.
        let f = p.parent()?.join(filename);
        return f.is_file().then(|| f.to_string_lossy().into_owned());
    }
    // Not a local path → HF Hub repo id (offline cache lookup).
    resolve_from_hub_cache(path, revision, filename)
}

/// Locate a file for an HF Hub repo id in the local cache. Offline —
/// the scheduler pre-downloads the model. `None` if not cached.
fn resolve_from_hub_cache(repo_id: &str, revision: Option<&str>, filename: &str) -> Option<String> {
    use hf_hub::{Cache, Repo, RepoType};

    // Python resolves the cache dir as HF_HUB_CACHE > HUGGINGFACE_HUB_CACHE >
    // HF_HOME/hub > ~/.cache/huggingface/hub; the hf-hub crate only knows
    // HF_HOME. Honor the explicit cache-dir overrides first, or the Rust
    // server misses models the Python scheduler already downloaded.
    let cache = ["HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"]
        .iter()
        .find_map(|var| std::env::var(var).ok())
        .map(|dir| Cache::new(dir.into()))
        .unwrap_or_else(Cache::from_env);

    let rev = revision.unwrap_or("main");
    cache
        .repo(Repo::with_revision(
            repo_id.to_string(),
            RepoType::Model,
            rev.to_string(),
        ))
        .get(filename)
        .map(|p| p.to_string_lossy().into_owned())
}

/// Real tokenizer over an already-loaded dynamo `Tokenizer` (Arc inside).
pub struct DynamoTokenizer {
    inner: dynamo_tokenizers::Tokenizer,
}

impl DynamoTokenizer {
    pub fn new(inner: dynamo_tokenizers::Tokenizer) -> Self {
        Self { inner }
    }
}

impl TextTokenizer for DynamoTokenizer {
    fn encode(&self, text: &str) -> Result<TokenIds, Error> {
        if text.is_empty() {
            // Match Python sglang: reject an empty prompt as a 400 (`Validation`),
            // not the misleading 500 a tokenize error would give.
            return Err(Error::Validation("prompt cannot be empty".into()));
        }
        let encoding = self
            .inner
            .encode(text)
            .map_err(|e| Error::Tokenize(e.to_string()))?;
        // Vocab ids are non-negative and fit in i32.
        Ok(encoding.token_ids().iter().map(|&id| id as i32).collect())
    }

    /// The post-processor prepends exactly what `encode("")` returns, so the
    /// probe is the same prefix [`strip_auto_specials`] removes.
    fn auto_specials(&self) -> Vec<i32> {
        self.inner
            .encode("")
            .map(|encoding| encoding.token_ids().iter().map(|&id| id as i32).collect())
            .unwrap_or_default()
    }
}

/// Remove one leading run of auto-added specials — exactly what an
/// `add_special_tokens=false` encode would have produced, without a second
/// tokenizer instance (the post-processor always prepends the same prefix, so
/// a template-rendered copy of those tokens is preserved).
fn strip_auto_specials(mut ids: Vec<i32>, auto_specials: &[i32]) -> Vec<i32> {
    if ids.starts_with(auto_specials) {
        ids.drain(..auto_specials.len());
    }
    ids
}

/// Probes for [`derive_affixes`]: unrelated strings that tokenize differently,
/// one ASCII and one not, so agreement between them is evidence the affixes are
/// constant rather than content-dependent.
//
// `allow(dead_code)`: only the `gigatoken` backend calls the affix machinery, and
// that feature cannot be enabled until gigatoken builds as a stable-Rust
// dependency (see the feature's comment in Cargo.toml). It stays compiled — and
// exercised by the unit tests below — rather than gated off, so the parity gate
// cannot rot while the dependency is blocked.
#[cfg_attr(not(feature = "gigatoken"), allow(dead_code))]
const AFFIX_PROBES: [&str; 3] = ["gigatoken affix probe", "下一个 42 probe", "a"];

/// Derive the `(prefix, suffix)` a reference tokenizer's post-processor puts
/// around a sequence, by diffing its output against a bare encoder's.
///
/// Used to teach an alternate backend — one whose `encode` applies no
/// post-processor, i.e. HuggingFace's `add_special_tokens=False` — what specials
/// the reference adds, so the two produce identical ids.
///
/// This doubles as the parity gate. Requiring the bare ids to appear *verbatim*
/// inside the reference output checks the whole content region token for token,
/// on every probe, so a vocabulary the alternate backend tokenizes differently
/// cannot reach the pool. `Err` means "do not use that backend for this model".
///
/// Ungated (and unit-tested) on purpose: this is the correctness-critical half
/// of an alternate encode backend, so it type-checks and runs in every build
/// regardless of which backends are compiled in.
#[cfg_attr(not(feature = "gigatoken"), allow(dead_code))]
fn derive_affixes(
    reference: &dyn TextTokenizer,
    mut bare_encode: impl FnMut(&str) -> Vec<i32>,
) -> Result<(Vec<i32>, Vec<i32>), String> {
    let mut resolved: Option<(Vec<i32>, Vec<i32>)> = None;
    for probe in AFFIX_PROBES {
        let bare = bare_encode(probe);
        if bare.is_empty() {
            return Err(format!(
                "alternate backend encoded probe {probe:?} to nothing"
            ));
        }
        let expected = reference
            .encode(probe)
            .map_err(|e| format!("reference tokenizer failed on probe {probe:?}: {e}"))?;
        let start = find_subslice(&expected, &bare).ok_or_else(|| {
            format!(
                "ids for probe {probe:?} do not appear in the reference tokenizer's \
                 output ({} ids vs {}); refusing this backend for this vocabulary",
                bare.len(),
                expected.len(),
            )
        })?;
        let affixes = (
            expected[..start].to_vec(),
            expected[start + bare.len()..].to_vec(),
        );
        match &resolved {
            None => resolved = Some(affixes),
            Some(first) if *first == affixes => {}
            Some(first) => {
                return Err(format!(
                    "special tokens are not a constant affix pair (probe {probe:?} \
                     implies {affixes:?}, an earlier probe implied {first:?})"
                ));
            }
        }
    }
    // AFFIX_PROBES is non-empty, so the loop always resolved.
    resolved.ok_or_else(|| "no affix probes configured".to_string())
}

/// Index where `needle` occurs contiguously in `haystack`.
#[cfg_attr(not(feature = "gigatoken"), allow(dead_code))]
fn find_subslice(haystack: &[i32], needle: &[i32]) -> Option<usize> {
    if needle.len() > haystack.len() {
        return None;
    }
    (0..=haystack.len() - needle.len()).find(|&i| &haystack[i..i + needle.len()] == needle)
}

/// One tokenizer worker: pulls a `Request` off the shared inbox, fills
/// `input_ids`, returns it to the TokenizerManager. Pinned; backend shared.
///
/// The `auto_specials` prefix (probed once at construction, Python's
/// `encode("")` probe) is stripped from template-rendered prompts —
/// [`GenerateRequest`]'s `skip_special_tokens` — so chat prompts gain no
/// extra BOS/EOS while native text keeps the post-processor specials.
pub struct TokenizerWorker {
    rx: flume::Receiver<Request>,
    tm: flume::Sender<TmEvent>,
    tokenizer: Arc<dyn TextTokenizer>,
    auto_specials: Vec<i32>,
}

impl TokenizerWorker {
    pub fn new(
        rx: flume::Receiver<Request>,
        tm: flume::Sender<TmEvent>,
        tokenizer: Arc<dyn TextTokenizer>,
    ) -> Self {
        let auto_specials = tokenizer.auto_specials();
        Self {
            rx,
            tm,
            tokenizer,
            auto_specials,
        }
    }
}

impl Runnable for TokenizerWorker {
    fn run(self) {
        while let Ok(mut req) = self.rx.recv() {
            // The tokenizer pool only ever receives generate requests. Encode,
            // then advance the FSM: `TokenizeDone` on success (→ PreSendValidating).
            let event = {
                let RequestKind::Generate(g) = &mut req.kind else {
                    tracing::error!("tokenizer pool received a non-generate request");
                    continue;
                };
                // Size the scheduler's stop-match window in TOKENS, as Python's
                // `normalize(tokenizer)` does.
                let stop_tokens = g
                    .sampling_params
                    .stop_strs
                    .iter()
                    // A stop that won't encode falls back to its byte length rather
                    // than failing the request: still an over-estimate, never an
                    // under-estimate, so the scheduler cannot miss that stop.
                    .map(|s| self.tokenizer.encode(s).map_or(s.len(), |ids| ids.len()))
                    .max();
                if let Some(n) = stop_tokens {
                    g.sampling_params.stop_str_max_len = n;
                }
                match self.tokenizer.encode(g.text.as_deref().unwrap_or("")) {
                    Ok(ids) => {
                        g.input_ids = Some(if g.skip_special_tokens {
                            strip_auto_specials(ids, &self.auto_specials)
                        } else {
                            ids
                        });
                        Event::TokenizeDone
                    }
                    Err(err) => Event::Error(err),
                }
            };
            let _ = req.state.apply(event);
            if self.tm.send(TmEvent::Tokenized(req)).is_err() {
                tracing::error!("tm inbox closed; dropping request");
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fsm::RequestState;
    use crate::message::{EgressSink, GenerateRequest, RequestKind, SamplingParams};
    use tokio::sync::mpsc;

    /// One token per whitespace-separated word, so a stop's token count differs
    /// from its byte count and the two units cannot be confused.
    struct WordTokenizer;
    impl TextTokenizer for WordTokenizer {
        fn encode(&self, text: &str) -> Result<TokenIds, Error> {
            Ok(text.split_whitespace().map(|_| 1i32).collect())
        }
    }

    /// The scheduler's stop-match window must reach the wire as a TOKEN count, as
    /// Python's `normalize(tokenizer)` produces.
    ///
    /// `Normalizing` leaves a UTF-8 BYTE count there — a safe over-estimate, but it
    /// makes the scheduler decode a longer tail on EVERY decode step of EVERY
    /// request (14 tokens vs 6 for a typical stop set). This stage owns the
    /// tokenizer, so it is where the exact count is resolved.
    #[test]
    fn tokenizing_replaces_the_byte_window_with_a_token_count() {
        let (req_tx, req_rx) = flume::unbounded::<Request>();
        let (tm_tx, tm_rx) = flume::unbounded::<TmEvent>();

        // 8 bytes vs 3 "tokens" under WordTokenizer — units are distinguishable.
        let sp = SamplingParams {
            stop_strs: vec!["a bb ccc".to_string(), "dd".to_string()],
            stop_str_max_len: 8, // what `normalize_stops` left: max BYTE length
            ..Default::default()
        };
        let (sink_tx, _sink_rx) = mpsc::channel(4);
        req_tx
            .send(Request {
                rid: "1".into(),
                state: RequestState::Tokenizing,
                sink: EgressSink::Local(sink_tx),
                kind: RequestKind::Generate(Box::new(GenerateRequest {
                    rid: "1".into(),
                    text: Some("hello world".into()),
                    sampling_params: sp,
                    ..Default::default()
                })),
            })
            .expect("send");
        drop(req_tx); // closes the loop after one request

        TokenizerWorker::new(req_rx, tm_tx, Arc::new(WordTokenizer)).run();

        let TmEvent::Tokenized(req) = tm_rx.try_recv().expect("returned") else {
            panic!("expected Tokenized");
        };
        let RequestKind::Generate(g) = &req.kind else {
            panic!("expected generate");
        };
        assert_eq!(
            g.sampling_params.stop_str_max_len, 3,
            "must be the max TOKEN count (3), not the byte count (8)"
        );
    }

    /// The strip reproduces `add_special_tokens=false`: one leading run of
    /// auto-added specials is removed, a template-rendered copy is kept, and
    /// tokenizers with no auto specials (empty probe) are untouched.
    #[test]
    fn strip_auto_specials_matches_add_special_tokens_false() {
        assert_eq!(strip_auto_specials(vec![0, 0, 1, 2], &[0]), vec![0, 1, 2]);
        assert_eq!(strip_auto_specials(vec![1, 2], &[0]), vec![1, 2]);
        assert_eq!(strip_auto_specials(vec![1, 2], &[]), vec![1, 2]);
        assert_eq!(strip_auto_specials(vec![0], &[0, 9]), vec![0]);
    }

    /// Word tokens plus a prepended BOS marker (id 0) — like an HF tokenizer
    /// whose post-processor adds specials.
    struct MarkedTokenizer;
    impl TextTokenizer for MarkedTokenizer {
        fn encode(&self, text: &str) -> Result<TokenIds, Error> {
            Ok(vec![0, text.len() as i32])
        }
        fn auto_specials(&self) -> Vec<i32> {
            vec![0]
        }
    }

    /// Reference that wraps one id per byte in BOS(0) … EOS(9) — an HF
    /// tokenizer whose post-processor adds specials on both sides.
    struct AffixedTokenizer;
    impl TextTokenizer for AffixedTokenizer {
        fn encode(&self, text: &str) -> Result<TokenIds, Error> {
            let mut ids = vec![0];
            ids.extend(text.bytes().map(|b| b as i32));
            ids.push(9);
            Ok(ids)
        }
    }

    /// The bare (`add_special_tokens=False`) counterpart of `AffixedTokenizer`.
    fn bare_bytes(text: &str) -> Vec<i32> {
        text.bytes().map(|b| b as i32).collect()
    }

    #[test]
    fn find_subslice_locates_the_content_region() {
        assert_eq!(find_subslice(&[0, 1, 2, 9], &[1, 2]), Some(1));
        assert_eq!(find_subslice(&[1, 2], &[1, 2]), Some(0));
        assert_eq!(find_subslice(&[1, 2], &[1, 2, 3]), None);
        assert_eq!(find_subslice(&[1, 2, 3], &[3, 2]), None);
    }

    /// Both affixes must be recovered, not just the prefix.
    ///
    /// The `auto_specials` probe (`encode("")`) can only see what a tokenizer
    /// PREPENDS. An alternate backend fed a prefix-only affix set would drop
    /// every prompt's trailing special — silently, since ids stay plausible.
    #[test]
    fn derive_affixes_recovers_prefix_and_suffix() {
        let (prefix, suffix) =
            derive_affixes(&AffixedTokenizer, bare_bytes).expect("constant affixes");
        assert_eq!(prefix, vec![0], "prefix");
        assert_eq!(suffix, vec![9], "suffix");
    }

    /// A backend whose ids differ from the reference must be REFUSED, not
    /// approximated. This is the whole safety argument for allowing an
    /// alternate encoder: a vocabulary it tokenizes differently never reaches
    /// the pool, so the flag cannot change what the model sees.
    #[test]
    fn derive_affixes_rejects_a_backend_that_tokenizes_differently() {
        // One id per CHARACTER, where the reference uses one per BYTE: identical
        // for ASCII, divergent as soon as a probe is non-ASCII.
        let by_char = |text: &str| text.chars().map(|c| c as i32).collect();
        let err =
            derive_affixes(&AffixedTokenizer, by_char).expect_err("divergent ids must be refused");
        assert!(
            err.contains("do not appear in the reference"),
            "expected a parity rejection, got: {err}"
        );
    }

    /// Specials that are not a constant affix pair (here: content-dependent)
    /// must be refused too — deriving from one probe and applying it to every
    /// request would corrupt prompts whose affixes differ.
    #[test]
    fn derive_affixes_rejects_content_dependent_specials() {
        /// Prepends one BOS per input byte, so probes of different lengths
        /// imply different affixes.
        struct VariableAffix;
        impl TextTokenizer for VariableAffix {
            fn encode(&self, text: &str) -> Result<TokenIds, Error> {
                let mut ids = vec![0; text.len()];
                ids.extend(text.bytes().map(|b| b as i32));
                Ok(ids)
            }
        }
        let err = derive_affixes(&VariableAffix, bare_bytes)
            .expect_err("content-dependent affixes must be refused");
        assert!(
            err.contains("not a constant affix pair"),
            "expected an affix-consistency rejection, got: {err}"
        );
    }

    /// A reference that cannot encode a probe must surface as an error rather
    /// than a silently empty affix set.
    #[test]
    fn derive_affixes_propagates_a_reference_failure() {
        struct Broken;
        impl TextTokenizer for Broken {
            fn encode(&self, _text: &str) -> Result<TokenIds, Error> {
                Err(Error::Tokenize("boom".into()))
            }
        }
        let err =
            derive_affixes(&Broken, bare_bytes).expect_err("must not be treated as no affixes");
        assert!(err.contains("reference tokenizer failed"), "got: {err}");
    }

    /// `skip_special_tokens` strips the probed prefix: template-rendered
    /// prompts (chat) must not gain a BOS the template didn't render — Python's
    /// `add_special_tokens=False` at the chat-template encode site.
    #[test]
    fn skip_special_tokens_strips_the_auto_added_specials() {
        let run = |skip_special_tokens: bool| {
            let (req_tx, req_rx) = flume::unbounded::<Request>();
            let (tm_tx, tm_rx) = flume::unbounded::<TmEvent>();
            req_tx
                .send(Request {
                    rid: "1".into(),
                    state: RequestState::Tokenizing,
                    sink: EgressSink::Local(tokio::sync::mpsc::channel(4).0),
                    kind: RequestKind::Generate(Box::new(GenerateRequest {
                        rid: "1".into(),
                        text: Some("hi".into()),
                        skip_special_tokens,
                        ..Default::default()
                    })),
                })
                .expect("send");
            drop(req_tx);
            TokenizerWorker::new(req_rx, tm_tx, Arc::new(MarkedTokenizer)).run();
            let TmEvent::Tokenized(req) = tm_rx.try_recv().expect("returned") else {
                panic!("expected Tokenized");
            };
            let RequestKind::Generate(g) = &req.kind else {
                panic!("expected generate");
            };
            g.input_ids.clone().expect("tokenized")
        };
        assert_eq!(run(false), vec![0, 2], "native prompts keep specials");
        assert_eq!(run(true), vec![2], "rendered prompts lose the auto BOS");
    }
}
