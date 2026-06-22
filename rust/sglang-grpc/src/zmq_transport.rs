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

    fn golden_req() -> TokenizedGenerateReqInput {
        build_tokenized_generate(
            Some("abc".into()),
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
}
