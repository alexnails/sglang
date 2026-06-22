//! ZMQ transport for the native generate path (RFC #22558 Phase 4).
//!
//! The Rust gRPC server pushes msgpack-encoded `TokenizedGenerateReqInput`s
//! straight to the Scheduler's input socket (`PortArgs.scheduler_input_ipc_name`),
//! bypassing the PyO3 bridge on the hot path. libzmq is used on both ends (the
//! Scheduler reads via pyzmq), so the bytes — already byte-identical to the
//! Python codec (see `msgpack::encode_tokenized_generate`) — interoperate.
//!
//! Request direction only for now; routing generated outputs back to the gRPC
//! server (the `http_worker_ipc` return path) is the remaining Phase-5 work.

use crate::msgpack;
use crate::proto::SamplingParams;
use crate::proto::ipc::{TokenArray, TokenizedGenerateReqInput};

#[derive(Debug)]
pub enum SendError {
    Encode(rmp_serde::encode::Error),
    Zmq(zmq::Error),
}

impl std::fmt::Display for SendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SendError::Encode(e) => write!(f, "msgpack encode failed: {e}"),
            SendError::Zmq(e) => write!(f, "zmq send failed: {e}"),
        }
    }
}
impl std::error::Error for SendError {}

impl From<rmp_serde::encode::Error> for SendError {
    fn from(e: rmp_serde::encode::Error) -> Self {
        SendError::Encode(e)
    }
}
impl From<zmq::Error> for SendError {
    fn from(e: zmq::Error) -> Self {
        SendError::Zmq(e)
    }
}

/// Build the Scheduler IPC request from a tokenized (input_ids) generate call.
/// `input_ids` is packed as native-endian int32, mirroring `array("i", ids)`.
pub fn build_tokenized_generate(
    rid: Option<String>,
    http_worker_ipc: Option<String>,
    input_text: Option<String>,
    input_ids: &[i32],
    sampling_params: Option<SamplingParams>,
    return_logprob: bool,
    logprob_start_len: i32,
    top_logprobs_num: i32,
    stream: bool,
) -> TokenizedGenerateReqInput {
    let data: Vec<u8> = input_ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    TokenizedGenerateReqInput {
        rid,
        // The Scheduler/Detokenizer routes this request's outputs back to the
        // endpoint a SchedulerReceiver is bound to (the Phase-5 return path).
        http_worker_ipc,
        input_text,
        input_ids: Some(TokenArray {
            typecode: "i".to_string(),
            data,
        }),
        sampling_params,
        return_logprob,
        logprob_start_len,
        top_logprobs_num,
        stream,
        ..Default::default()
    }
}

/// A PUSH socket connected to the Scheduler's input endpoint. libzmq sends are
/// synchronous; callers on an async runtime should use `spawn_blocking`.
pub struct SchedulerSender {
    _ctx: zmq::Context,
    socket: zmq::Socket,
}

impl SchedulerSender {
    /// Connect a PUSH socket to `endpoint` (e.g. `ipc:///tmp/...` or `tcp://...`),
    /// matching `PortArgs.scheduler_input_ipc_name` on the Python side.
    pub fn connect(endpoint: &str) -> zmq::Result<Self> {
        let ctx = zmq::Context::new();
        let socket = ctx.socket(zmq::PUSH)?;
        // Flush pending sends for up to 2s on close instead of dropping them
        // (libzmq's default of discard-on-close would silently lose a request
        // pushed just before the socket/context is torn down).
        socket.set_linger(2000)?;
        socket.connect(endpoint)?;
        Ok(Self { _ctx: ctx, socket })
    }

    /// Encode and push a tokenized generate request to the Scheduler.
    pub fn send_tokenized_generate(
        &self,
        req: &TokenizedGenerateReqInput,
    ) -> Result<(), SendError> {
        let bytes = msgpack::encode_tokenized_generate(req)?;
        self.socket.send(bytes, 0)?;
        Ok(())
    }
}

/// One request's slice of a `BatchStrOutput` (the gRPC server streams these).
#[derive(Debug, Clone, PartialEq)]
pub struct GenerateOutput {
    pub rid: Option<String>,
    pub output_text: String,
    pub output_ids: Vec<i32>,
    pub finished: bool,
}

fn token_array_to_ids(v: &rmpv::Value) -> Vec<i32> {
    // io_struct sends array.array as [typecode, raw_bytes]; we only need the ids.
    if let Some(pair) = v.as_array() {
        if pair.len() == 2 {
            if let rmpv::Value::Binary(data) = &pair[1] {
                return data
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
            }
        }
    }
    Vec::new()
}

/// Decode the leading positional fields of a `BatchStrOutput` (array_like+tag).
/// We read only what the gRPC server streams: rid, output text/ids, finished.
/// Field order (after the tag): rids, http_worker_ipcs, finished_reasons,
/// output_strs, output_ids, ...
pub fn decode_batch_str_output(bytes: &[u8]) -> Result<Vec<GenerateOutput>, String> {
    let value = rmpv::decode::read_value(&mut &bytes[..]).map_err(|e| e.to_string())?;
    let arr = value.as_array().ok_or("BatchStrOutput: expected an array")?;
    if arr.len() < 6 {
        return Err(format!("BatchStrOutput: short array (len {})", arr.len()));
    }
    let rids = arr[1].as_array();
    let finished = arr[3].as_array().ok_or("finished_reasons: expected array")?;
    let strs = arr[4].as_array().ok_or("output_strs: expected array")?;
    let ids = arr[5].as_array();

    let mut out = Vec::with_capacity(strs.len());
    for (i, s) in strs.iter().enumerate() {
        out.push(GenerateOutput {
            rid: rids
                .and_then(|r| r.get(i))
                .and_then(|x| x.as_str())
                .map(str::to_string),
            output_text: s.as_str().unwrap_or_default().to_string(),
            output_ids: ids
                .and_then(|l| l.get(i))
                .map(token_array_to_ids)
                .unwrap_or_default(),
            // A non-nil finish reason means this request is done.
            finished: finished.get(i).map(|fr| !fr.is_nil()).unwrap_or(false),
        });
    }
    Ok(out)
}

/// A PULL socket the gRPC server binds for generated outputs; the request's
/// `http_worker_ipc` points the Scheduler/Detokenizer here (Phase-5 return path).
pub struct SchedulerReceiver {
    _ctx: zmq::Context,
    socket: zmq::Socket,
}

impl SchedulerReceiver {
    pub fn bind(endpoint: &str) -> zmq::Result<Self> {
        let ctx = zmq::Context::new();
        let socket = ctx.socket(zmq::PULL)?;
        socket.bind(endpoint)?;
        Ok(Self { _ctx: ctx, socket })
    }

    pub fn set_recv_timeout_ms(&self, ms: i32) -> zmq::Result<()> {
        self.socket.set_rcvtimeo(ms)
    }

    /// Receive one output frame and decode its per-request slices.
    pub fn recv_outputs(&self) -> Result<Vec<GenerateOutput>, String> {
        let bytes = self.socket.recv_bytes(0).map_err(|e| e.to_string())?;
        decode_batch_str_output(&bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn from_hex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    // Same float-free vector as msgpack::tests + test_msgpack_codec.py.
    const GV_TOK_GENERATE: &str = "9eb9546f6b656e697a656447656e6572617465526571496e707574a3616263c0a2686992a169c40c010000000200000003000000c0c0c083ae6d61785f6e65775f746f6b656e7310ae73746f705f746f6b656e5f6964739102a16e01c2000090c3";
    // A faithful 1-item BatchStrOutput minted from io_struct (scratchpad/mock_scheduler.py mint):
    // output_strs=["hello world"], output_ids=[array("i",[10,11,12])], finish=stop.
    const GV_BATCH_STR_OUTPUT: &str = "dc0028ae42617463685374724f757470757491a361626391ac6970633a2f2f2f746d702f789182a474797065a473746f70a76d617463686564c091ab68656c6c6f20776f726c649192a169c40c0a0000000b0000000c000000910391039100910090909090909090909090909090909090909090c0c0c0c0c0c0c0c0909090";

    fn golden_req() -> TokenizedGenerateReqInput {
        build_tokenized_generate(
            Some("abc".into()),
            None, // http_worker_ipc
            Some("hi".into()),
            &[1, 2, 3],
            Some(SamplingParams {
                max_new_tokens: Some(16),
                n: Some(1),
                stop_token_ids: vec![2],
                ..Default::default()
            }),
            false,
            0,
            0,
            true,
        )
    }

    #[test]
    fn builder_matches_golden_vector() {
        assert_eq!(
            msgpack::encode_tokenized_generate(&golden_req()).unwrap(),
            from_hex(GV_TOK_GENERATE)
        );
    }

    #[test]
    #[ignore = "manual Rust->Python interop demo; needs SGL_ZMQ_DEMO_ENDPOINT + a pyzmq PULL"]
    fn interop_push_to_python_endpoint() {
        let endpoint = std::env::var("SGL_ZMQ_DEMO_ENDPOINT")
            .expect("set SGL_ZMQ_DEMO_ENDPOINT to the Python PULL endpoint");
        let sender = SchedulerSender::connect(&endpoint).unwrap();
        sender.send_tokenized_generate(&golden_req()).unwrap();
        eprintln!("pushed golden TokenizedGenerateReqInput -> {endpoint}");
        // Give libzmq a moment to deliver before the process tears down.
        std::thread::sleep(std::time::Duration::from_millis(500));
    }

    #[test]
    fn push_pull_carries_exact_bytes_over_real_zmq() {
        // Real libzmq PUSH->PULL over an ipc socket: the exact golden bytes must
        // arrive intact (transitively: a pyzmq PULL + io_struct.msgpack_decode
        // would reconstruct the request, since the bytes are byte-identical).
        let dir = std::env::temp_dir().join(format!("sgl-zmq-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let endpoint = format!("ipc://{}/sched.ipc", dir.display());

        let ctx = zmq::Context::new();
        let pull = ctx.socket(zmq::PULL).unwrap();
        pull.set_rcvtimeo(5000).unwrap();
        pull.bind(&endpoint).unwrap();

        let sender = SchedulerSender::connect(&endpoint).unwrap();
        sender.send_tokenized_generate(&golden_req()).unwrap();

        let received = pull.recv_bytes(0).unwrap();
        assert_eq!(received, from_hex(GV_TOK_GENERATE));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn decode_batch_str_output_reads_leading_fields() {
        let outs = decode_batch_str_output(&from_hex(GV_BATCH_STR_OUTPUT)).unwrap();
        assert_eq!(outs.len(), 1);
        assert_eq!(outs[0].rid.as_deref(), Some("abc"));
        assert_eq!(outs[0].output_text, "hello world");
        assert_eq!(outs[0].output_ids, vec![10, 11, 12]);
        assert!(outs[0].finished);
    }

    #[test]
    #[ignore = "manual E2E vs scratchpad/mock_scheduler.py; set SGL_REQ_ENDPOINT + SGL_OUT_ENDPOINT"]
    fn e2e_roundtrip_with_mock_scheduler() {
        let req_ep = std::env::var("SGL_REQ_ENDPOINT").expect("set SGL_REQ_ENDPOINT");
        let out_ep = std::env::var("SGL_OUT_ENDPOINT").expect("set SGL_OUT_ENDPOINT");
        // Bind our output receiver BEFORE pushing, so the reply isn't missed.
        let receiver = SchedulerReceiver::bind(&out_ep).unwrap();
        receiver.set_recv_timeout_ms(15000).unwrap();
        let sender = SchedulerSender::connect(&req_ep).unwrap();
        let req = build_tokenized_generate(
            Some("abc".into()),
            Some(out_ep.clone()), // route outputs back to our receiver
            Some("hi".into()),
            &[1, 2, 3],
            Some(SamplingParams {
                max_new_tokens: Some(16),
                ..Default::default()
            }),
            false,
            0,
            0,
            true,
        );
        sender.send_tokenized_generate(&req).unwrap();
        let outs = receiver.recv_outputs().unwrap();
        eprintln!("E2E received: {outs:?}");
        assert_eq!(outs.len(), 1);
        assert_eq!(outs[0].output_text, "hello world");
        assert_eq!(outs[0].output_ids, vec![10, 11, 12]);
        assert!(outs[0].finished);
    }
}
