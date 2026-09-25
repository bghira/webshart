use super::*;
use std::io::Write;

const FILE_HASH: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const XORB_HASH: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

fn test_file(server: &mockito::Server, size: u64) -> XetFile {
    XetFile {
        hash: FILE_HASH.into(),
        size,
        sha256: None,
        auth_url: Url::parse(&format!("{}/auth", server.url())).unwrap(),
        hub_token: Some("hub-secret".into()),
        credentials: AsyncMutex::new(Some(Credentials {
            access_token: "old-cas-token".into(),
            exp: 4102444800,
            cas_url: Url::parse(&server.url()).unwrap(),
        })),
    }
}

fn recipe(server: &mockito::Server, data: &[u8], encoded: &[u8], offset: u64) -> serde_json::Value {
    serde_json::json!({
        "offset_into_first_range": offset,
        "terms": [{"hash": XORB_HASH, "unpacked_length": data.len(), "range": {"start": 0, "end": 1}}],
        "xorbs": {XORB_HASH: [{"url": format!("{}/xorb", server.url()), "ranges": [{"chunks": {"start": 0, "end": 1}, "bytes": {"start": 0, "end": encoded.len() - 1}}]}]}
    })
}

fn encode_chunk(data: &[u8], compression: u8) -> Vec<u8> {
    let grouped: Vec<u8> = if compression == 2 {
        (0..4)
            .flat_map(|lane| data.iter().skip(lane).step_by(4).copied())
            .collect()
    } else {
        data.to_vec()
    };
    let encoded = if compression == 0 {
        grouped
    } else {
        let mut encoder = lz4_flex::frame::FrameEncoder::new(Vec::new());
        encoder.write_all(&grouped).unwrap();
        encoder.finish().unwrap()
    };
    let mut output = vec![0];
    output.extend_from_slice(&(encoded.len() as u32).to_le_bytes()[..3]);
    output.push(compression);
    output.extend_from_slice(&(data.len() as u32).to_le_bytes()[..3]);
    output.extend_from_slice(&encoded);
    output
}

#[test]
fn all_chunk_encodings_round_trip_with_incomplete_byte_groups() {
    for length in [1, 2, 3, 4, 5, 1023, 1024, 1025, 128 * 1024] {
        let data: Vec<_> = (0..length).map(|index| (index % 251) as u8).collect();
        for compression in 0..=2 {
            let payload = encode_chunk(&data, compression);
            let decoded = format::decode_chunks(
                &payload,
                &Interval { start: 5, end: 6 },
                &HashSet::from([5]),
            )
            .unwrap();
            assert_eq!(decoded[&5], data);
        }
    }
}

#[test]
fn rejects_invalid_and_truncated_chunks() {
    let payload = encode_chunk(b"sample bytes", 1);
    for length in 0..payload.len() {
        assert!(format::decode_chunks(
            &payload[..length],
            &Interval { start: 0, end: 1 },
            &HashSet::from([0])
        )
        .is_err());
    }
    for position in [0, 4, 5, 6, 7] {
        let mut corrupted = payload.clone();
        corrupted[position] = 255;
        assert!(format::decode_chunks(
            &corrupted,
            &Interval { start: 0, end: 1 },
            &HashSet::from([0])
        )
        .is_err());
    }
}

#[test]
fn pooled_connections_work_across_independent_runtimes() {
    let mut server = mockito::Server::new();
    let response = server
        .mock("GET", "/runtime")
        .with_body("ok")
        .expect(2)
        .create();
    let client = XetClient::new(&server.url(), 2).unwrap();
    let url = Url::parse(&format!("{}/runtime", server.url())).unwrap();
    let first = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let second = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    for runtime in [&first, &second] {
        runtime.block_on(async {
            let reply = tokio::time::timeout(
                Duration::from_secs(3),
                client.request(Method::GET, &url, None, None, 2),
            )
            .await
            .expect("Xet request stalled on an idle runtime")
            .unwrap();
            assert_eq!(reply.body, b"ok");
        });
    }
    response.assert();
}

#[test]
fn multipart_ranges_preserve_binary_data_and_reject_mismatches() {
    let ranges = vec![
        FetchRange {
            chunks: Interval { start: 0, end: 1 },
            bytes: Interval { start: 10, end: 12 },
        },
        FetchRange {
            chunks: Interval { start: 2, end: 3 },
            bytes: Interval { start: 20, end: 22 },
        },
    ];
    let body = b"--test\r\nContent-Range: bytes 20-22/100\r\n\r\nxyz\r\n--test\r\nContent-Range: bytes 10-12/100\r\n\r\n\0\xffa\r\n--test--\r\n";
    let mut headers = HeaderMap::new();
    headers.insert(
        "content-type",
        "multipart/byteranges; boundary=\"test\"".parse().unwrap(),
    );
    let parts = format::parts(body, &headers, &ranges).unwrap();
    assert_eq!(parts[0].1, b"xyz");
    assert_eq!(parts[1].1, b"\0\xffa");
    for preamble in [b"\r\n".as_slice(), b"MIME preamble\r\n"] {
        let prefixed = [preamble, body].concat();
        assert_eq!(
            format::parts(&prefixed, &headers, &ranges).unwrap()[0].1,
            b"xyz"
        );
    }
    assert!(format::parts(&body[..body.len() - 15], &headers, &ranges).is_err());
    assert!(format::parts(body, &headers, &ranges[..1]).is_err());
}

#[tokio::test]
async fn native_download_and_partial_read_with_token_exchange() {
    let mut server = mockito::Server::new_async().await;
    let data = b"structured captions and image bytes";
    let payload = encode_chunk(data, 2);
    let route = "/datasets/test/data/resolve/main/shard.tar";
    let probe = server
        .mock("HEAD", route)
        .match_header("authorization", "Bearer hub-secret")
        .with_status(302)
        .with_header("x-xet-hash", FILE_HASH)
        .with_header("x-linked-size", &data.len().to_string())
        .with_header("x-linked-etag", &digest_to_hex(Sha256::digest(data)))
        .with_header("location", "/must-not-follow")
        .expect(1)
        .create_async()
        .await;
    let auth = server.mock("GET", "/api/datasets/test/data/xet-read-token/main")
        .match_header("authorization", "Bearer hub-secret")
        .with_body(serde_json::json!({"accessToken": "cas-secret", "exp": 4102444800u64, "casUrl": server.url()}).to_string())
        .expect(1).create_async().await;
    let reconstruction = server.mock("GET", format!("/v2/reconstructions/{FILE_HASH}").as_str())
        .match_header("authorization", "Bearer cas-secret")
        .with_body(serde_json::json!({
            "offset_into_first_range": 0,
            "terms": [{"hash": XORB_HASH, "unpacked_length": data.len(), "range": {"start": 0, "end": 1}}],
            "xorbs": {XORB_HASH: [{"url": format!("{}/xorb?signed=secret", server.url()), "ranges": [{"chunks": {"start": 0, "end": 1}, "bytes": {"start": 0, "end": payload.len() - 1}}]}]}
        }).to_string()).expect(2).create_async().await;
    let transfer = server
        .mock("GET", "/xorb?signed=secret")
        .match_header("authorization", mockito::Matcher::Missing)
        .match_header("range", format!("bytes=0-{}", payload.len() - 1).as_str())
        .with_status(206)
        .with_header(
            "content-range",
            &format!("bytes 0-{}/{}", payload.len() - 1, payload.len()),
        )
        .with_body(payload)
        .expect(2)
        .create_async()
        .await;
    let client = XetClient::new(&server.url(), 4).unwrap();
    let file = client
        .resolve(&format!("{}{route}", server.url()), Some("hub-secret"))
        .await
        .unwrap()
        .unwrap();
    assert!(client
        .resolve(&format!("{}{route}", server.url()), Some("hub-secret"))
        .await
        .unwrap()
        .is_some());
    assert_eq!(client.read_range(&file, 0, 10).await.unwrap(), &data[..10]);
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("shard.tar");
    assert_eq!(
        client.download(&file, &path).await.unwrap(),
        data.len() as u64
    );
    assert_eq!(std::fs::read(path).unwrap(), data);
    probe.assert_async().await;
    auth.assert_async().await;
    reconstruction.assert_async().await;
    transfer.assert_async().await;
}

#[test]
fn auth_routes_only_accept_the_configured_hub_and_preserve_encoded_revisions() {
    let client = XetClient::new("https://huggingface.co", 4).unwrap();
    for (path, expected) in [
        (
            "/datasets/org/data/resolve/refs%2Fpr%2F1/data/a.tar",
            "/api/datasets/org/data/xet-read-token/refs%2Fpr%2F1",
        ),
        (
            "/org/model/resolve/main/weights.bin",
            "/api/models/org/model/xet-read-token/main",
        ),
        (
            "/spaces/org/demo/resolve/main/a.tar",
            "/api/spaces/org/demo/xet-read-token/main",
        ),
    ] {
        let url = Url::parse(&format!("https://huggingface.co{path}")).unwrap();
        assert_eq!(client.auth_url(&url, None).unwrap().path(), expected);
        assert!(client
            .auth_url(&url, Some("commit-id"))
            .unwrap()
            .path()
            .ends_with("/commit-id"));
    }
    for url in [
        "https://huggingface.co.attacker.test/datasets/org/data/resolve/main/a.tar",
        "https://example.com/a.tar",
        "https://huggingface.co/org/model/blob/main/a.tar",
    ] {
        assert!(client.auth_url(&Url::parse(url).unwrap(), None).is_none());
    }
}

#[tokio::test]
async fn non_xet_files_keep_http_and_probe_cache_is_auth_scoped() {
    let mut server = mockito::Server::new_async().await;
    let route = "/datasets/org/data/resolve/main/a.tar";
    let plain = server
        .mock("HEAD", route)
        .with_status(302)
        .expect(2)
        .create_async()
        .await;
    let client = XetClient::new(&server.url(), 2).unwrap();
    for token in [None, None, Some("separate-user")] {
        assert!(client
            .resolve(&format!("{}{route}", server.url()), token)
            .await
            .unwrap()
            .is_none());
    }
    plain.assert_async().await;
    let denied = server
        .mock("HEAD", route)
        .with_status(403)
        .expect(1)
        .create_async()
        .await;
    assert!(matches!(
        client
            .resolve(&format!("{}{route}", server.url()), Some("denied-user"))
            .await,
        Err(WebshartError::XetHttp(403))
    ));
    denied.assert_async().await;
}

#[tokio::test]
async fn partial_ranges_trim_chunks_and_reject_truncated_terms() {
    let mut server = mockito::Server::new_async().await;
    let data = b"0123456789abcdefghij";
    let payload = encode_chunk(data, 1);
    let response = recipe(&server, data, &payload, 7);
    let reconstruction = server
        .mock("GET", format!("/v2/reconstructions/{FILE_HASH}").as_str())
        .match_header("range", "bytes=7-12")
        .with_body(response.to_string())
        .create_async()
        .await;
    let transfer = server
        .mock("GET", "/xorb")
        .with_status(206)
        .with_header(
            "content-range",
            &format!("bytes 0-{}/{}", payload.len() - 1, payload.len()),
        )
        .with_body(payload)
        .expect(2)
        .create_async()
        .await;
    let client = XetClient::new(&server.url(), 2).unwrap();
    let file = test_file(&server, data.len() as u64);
    assert_eq!(client.read_range(&file, 7, 6).await.unwrap(), b"789abc");
    reconstruction.assert_async().await;
    let mut corrupted = response;
    corrupted["terms"][0]["unpacked_length"] = serde_json::json!(data.len() + 1);
    assert!(client
        .assemble(serde_json::from_value(corrupted).unwrap(), 6)
        .await
        .unwrap_err()
        .to_string()
        .contains("length mismatch"));
    transfer.assert_async().await;
    assert!(client.read_range(&file, 19, 2).await.is_err());
    assert!(client.read_range(&file, u64::MAX, 2).await.is_err());
    assert!(client.read_range(&file, 20, 0).await.unwrap().is_empty());
}

#[tokio::test]
async fn duplicate_terms_reuse_chunks_and_reconstruct_in_term_order() {
    let mut server = mockito::Server::new_async().await;
    let mut payload = encode_chunk(b"abc", 0);
    payload.extend(encode_chunk(b"XYZ", 1));
    let response = serde_json::json!({
        "offset_into_first_range": 1,
        "terms": [
            {"hash": XORB_HASH, "unpacked_length": 3, "range": {"start": 1, "end": 2}},
            {"hash": XORB_HASH, "unpacked_length": 6, "range": {"start": 0, "end": 2}}
        ],
        "xorbs": {XORB_HASH: [{"url": format!("{}/xorb", server.url()), "ranges": [{"chunks": {"start": 0, "end": 2}, "bytes": {"start": 0, "end": payload.len() - 1}}]}]}
    });
    let transfer = server
        .mock("GET", "/xorb")
        .with_status(206)
        .with_header(
            "content-range",
            &format!("bytes 0-{}/{}", payload.len() - 1, payload.len()),
        )
        .with_body(payload)
        .expect(1)
        .create_async()
        .await;
    let client = XetClient::new(&server.url(), 2).unwrap();
    assert_eq!(
        client
            .assemble(serde_json::from_value(response).unwrap(), 7)
            .await
            .unwrap(),
        b"YZabcXY"
    );
    transfer.assert_async().await;
}

#[tokio::test]
async fn v1_fallback_and_transient_request_retry() {
    let mut server = mockito::Server::new_async().await;
    let data = b"legacy data";
    let payload = encode_chunk(data, 0);
    let version2 = server
        .mock("GET", format!("/v2/reconstructions/{FILE_HASH}").as_str())
        .with_status(501)
        .expect(1)
        .create_async()
        .await;
    let version1 = server.mock("GET", format!("/v1/reconstructions/{FILE_HASH}").as_str())
        .with_body(serde_json::json!({
            "offset_into_first_range": 0,
            "terms": [{"hash": XORB_HASH, "unpacked_length": data.len(), "range": {"start": 0, "end": 1}}],
            "fetch_info": {XORB_HASH: [{"url": format!("{}/xorb", server.url()), "range": {"start": 0, "end": 1}, "url_range": {"start": 0, "end": payload.len() - 1}}]}
        }).to_string()).create_async().await;
    let transient = server
        .mock("GET", "/xorb")
        .with_status(503)
        .with_header("retry-after", "0")
        .expect(1)
        .create_async()
        .await;
    let transfer = server
        .mock("GET", "/xorb")
        .with_status(206)
        .with_header(
            "content-range",
            &format!("bytes 0-{}/{}", payload.len() - 1, payload.len()),
        )
        .with_body(payload)
        .create_async()
        .await;
    let client = XetClient::new(&server.url(), 2).unwrap();
    assert_eq!(
        client
            .read_range(&test_file(&server, data.len() as u64), 0, data.len() as u64)
            .await
            .unwrap(),
        data
    );
    version2.assert_async().await;
    version1.assert_async().await;
    transient.assert_async().await;
    transfer.assert_async().await;
}

#[tokio::test]
async fn expired_signed_urls_refresh_reconstruction_and_tokens_without_leaking_auth() {
    let mut server = mockito::Server::new_async().await;
    let data = b"fresh data";
    let payload = encode_chunk(data, 1);
    let mut stale = recipe(&server, data, &payload, 0);
    stale["xorbs"][XORB_HASH][0]["url"] =
        serde_json::json!(format!("{}/expired?secret=do-not-print", server.url()));
    let old = server
        .mock("GET", format!("/v2/reconstructions/{FILE_HASH}").as_str())
        .match_header("authorization", "Bearer old-cas-token")
        .with_body(stale.to_string())
        .expect(1)
        .create_async()
        .await;
    let expired = server
        .mock("GET", "/expired?secret=do-not-print")
        .match_header("authorization", mockito::Matcher::Missing)
        .with_status(403)
        .expect(1)
        .create_async()
        .await;
    let auth = server.mock("GET", "/auth").match_header("authorization", "Bearer hub-secret")
        .with_body(serde_json::json!({"accessToken": "new-cas-token", "exp": 4102444800u64, "casUrl": server.url()}).to_string()).create_async().await;
    let fresh = server
        .mock("GET", format!("/v2/reconstructions/{FILE_HASH}").as_str())
        .match_header("authorization", "Bearer new-cas-token")
        .with_body(recipe(&server, data, &payload, 0).to_string())
        .create_async()
        .await;
    let transfer = server
        .mock("GET", "/xorb")
        .match_header("authorization", mockito::Matcher::Missing)
        .with_status(206)
        .with_header(
            "content-range",
            &format!("bytes 0-{}/{}", payload.len() - 1, payload.len()),
        )
        .with_body(payload)
        .create_async()
        .await;
    let client = XetClient::new(&server.url(), 2).unwrap();
    assert_eq!(
        client
            .read_range(&test_file(&server, data.len() as u64), 0, data.len() as u64)
            .await
            .unwrap(),
        data
    );
    for mock in [&old, &expired, &auth, &fresh, &transfer] {
        mock.assert_async().await;
    }
}

#[tokio::test]
async fn corrupt_full_download_is_rejected_by_sha256() {
    let mut server = mockito::Server::new_async().await;
    let data = b"corrupted";
    let payload = encode_chunk(data, 0);
    let reconstruction = server
        .mock("GET", format!("/v2/reconstructions/{FILE_HASH}").as_str())
        .with_body(recipe(&server, data, &payload, 0).to_string())
        .create_async()
        .await;
    let transfer = server
        .mock("GET", "/xorb")
        .with_status(206)
        .with_header(
            "content-range",
            &format!("bytes 0-{}/{}", payload.len() - 1, payload.len()),
        )
        .with_body(payload)
        .create_async()
        .await;
    let client = XetClient::new(&server.url(), 2).unwrap();
    let mut file = test_file(&server, data.len() as u64);
    file.sha256 = Some(digest_to_hex(Sha256::digest(b"expected!")));
    let directory = tempfile::tempdir().unwrap();
    assert!(client
        .download(&file, &directory.path().join("shard.part"))
        .await
        .unwrap_err()
        .to_string()
        .contains("SHA-256 mismatch"));
    reconstruction.assert_async().await;
    transfer.assert_async().await;
}

#[tokio::test]
async fn transfer_errors_do_not_expose_signed_urls() {
    let mut server = mockito::Server::new_async().await;
    let error = server
        .mock("GET", "/bad?Signature=secret")
        .with_status(400)
        .create_async()
        .await;
    let client = XetClient::new(&server.url(), 2).unwrap();
    let fetch = Fetch {
        url: Url::parse(&format!("{}/bad?Signature=secret", server.url())).unwrap(),
        ranges: vec![FetchRange {
            chunks: Interval { start: 0, end: 1 },
            bytes: Interval { start: 0, end: 10 },
        }],
    };
    let message = client
        .fetch(&fetch, &HashSet::from([0]))
        .await
        .unwrap_err()
        .to_string();
    assert!(message.contains("400"));
    assert!(!message.contains("Signature") && !message.contains("secret"));
    error.assert_async().await;
}

#[tokio::test]
#[ignore = "downloads a public 287 MiB shard and checks independent HTTP ranges"]
async fn live_public_hub_shard() {
    let url = "https://huggingface.co/datasets/webshart/pseudo-camera-10k-structured/resolve/3ad79f91c50aab9dc63cbae9daf2ec2c474c34af/data/shard-00018.tar";
    let client = XetClient::new("https://huggingface.co", 8).unwrap();
    let file = client
        .resolve(url, None)
        .await
        .unwrap()
        .expect("Expected native Xet metadata");
    assert_eq!(file.size, 301424640);
    assert_eq!(
        file.sha256.as_deref(),
        Some("990c661ebaa713d3e5a0c76f781e9bc6cff22e4273a1d09228283ab270ce79d6")
    );
    let http = Client::new();
    for (offset, length) in [
        (37, 150000),
        (WINDOW_BYTES - 17, 65539),
        (file.size - 2048, 2048),
    ] {
        let native = client.read_range(&file, offset, length).await.unwrap();
        let response = http
            .get(url)
            .header("Range", format!("bytes={offset}-{}", offset + length - 1))
            .send()
            .await
            .map_err(|error| error.without_url())
            .unwrap();
        assert_eq!(response.status(), StatusCode::PARTIAL_CONTENT);
        let reference = response
            .bytes()
            .await
            .map_err(|error| error.without_url())
            .unwrap();
        assert_eq!(native, reference);
    }
    let directory = tempfile::tempdir().unwrap();
    let started = Instant::now();
    assert_eq!(
        client
            .download(&file, &directory.path().join("shard.tar"))
            .await
            .unwrap(),
        file.size
    );
    println!(
        "Native Xet: {} bytes downloaded and SHA-256 verified in {:.2}s",
        file.size,
        started.elapsed().as_secs_f64()
    );
}

#[tokio::test]
async fn xorb_transfers_are_parallel_and_respect_the_concurrency_limit() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base_url = format!("http://{}", listener.local_addr().unwrap());
    let active = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let encoded = encode_chunk(b"x", 0);
    let encoded_length = encoded.len();
    let server_peak = peak.clone();
    let server = tokio::spawn(async move {
        let mut handlers = tokio::task::JoinSet::new();
        for _ in 0..8 {
            let (mut socket, _) = listener.accept().await.unwrap();
            let active = active.clone();
            let peak = server_peak.clone();
            let encoded = encoded.clone();
            handlers.spawn(async move {
                let mut request = Vec::new();
                while !request.ends_with(b"\r\n\r\n") {
                    request.push(socket.read_u8().await.unwrap());
                }
                let count = active.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(count, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(25)).await;
                let response = format!("HTTP/1.1 206 Partial Content\r\nContent-Length: {encoded_length}\r\nContent-Range: bytes 0-{}/{encoded_length}\r\nConnection: close\r\n\r\n", encoded_length - 1);
                socket.write_all(response.as_bytes()).await.unwrap();
                socket.write_all(&encoded).await.unwrap();
                active.fetch_sub(1, Ordering::SeqCst);
            });
        }
        while let Some(result) = handlers.join_next().await {
            result.unwrap();
        }
    });
    let client = XetClient::new(&base_url, 3).unwrap();
    let terms = (0..8)
        .map(|index| Term {
            hash: format!("{index:064x}"),
            unpacked_length: 1,
            range: Interval { start: 0, end: 1 },
        })
        .collect();
    let xorbs = (0..8)
        .map(|index| {
            (
                format!("{index:064x}"),
                vec![Fetch {
                    url: Url::parse(&format!("{base_url}/xorb/{index}")).unwrap(),
                    ranges: vec![FetchRange {
                        chunks: Interval { start: 0, end: 1 },
                        bytes: Interval {
                            start: 0,
                            end: encoded_length as u64 - 1,
                        },
                    }],
                }],
            )
        })
        .collect();
    let reconstruction = Reconstruction {
        offset_into_first_range: 0,
        terms,
        xorbs,
        fetch_info: HashMap::new(),
    };
    let output = tokio::time::timeout(Duration::from_secs(5), client.assemble(reconstruction, 8))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(output, b"xxxxxxxx");
    assert_eq!(peak.load(Ordering::SeqCst), 3);
    server.await.unwrap();
}

#[tokio::test]
async fn signed_multi_range_fetches_decode_nonadjacent_chunks() {
    let mut server = mockito::Server::new_async().await;
    let first = encode_chunk(b"first", 1);
    let second = encode_chunk(b"second", 2);
    let mut body = format!(
        "--boundary\r\nContent-Range: bytes 0-{}/2000\r\n\r\n",
        first.len() - 1
    )
    .into_bytes();
    body.extend_from_slice(&first);
    body.extend_from_slice(
        format!(
            "\r\n--boundary\r\nContent-Range: bytes 1000-{}/2000\r\n\r\n",
            1000 + second.len() - 1
        )
        .as_bytes(),
    );
    body.extend_from_slice(&second);
    body.extend_from_slice(b"\r\n--boundary--\r\n");
    let transfer = server
        .mock("GET", "/multi?signed=ranges")
        .match_header(
            "range",
            format!(
                "bytes=0-{},1000-{}",
                first.len() - 1,
                1000 + second.len() - 1
            )
            .as_str(),
        )
        .match_header("authorization", mockito::Matcher::Missing)
        .with_status(206)
        .with_header("content-type", "multipart/byteranges; boundary=boundary")
        .with_body(body)
        .create_async()
        .await;
    let client = XetClient::new(&server.url(), 2).unwrap();
    let fetch = Fetch {
        url: Url::parse(&format!("{}/multi?signed=ranges", server.url())).unwrap(),
        ranges: vec![
            FetchRange {
                chunks: Interval { start: 0, end: 1 },
                bytes: Interval {
                    start: 0,
                    end: first.len() as u64 - 1,
                },
            },
            FetchRange {
                chunks: Interval { start: 2, end: 3 },
                bytes: Interval {
                    start: 1000,
                    end: 1000 + second.len() as u64 - 1,
                },
            },
        ],
    };
    let chunks = client.fetch(&fetch, &HashSet::from([0, 2])).await.unwrap();
    assert_eq!(chunks[&0], b"first");
    assert_eq!(chunks[&2], b"second");
    transfer.assert_async().await;
}

#[tokio::test]
async fn chunk_ranges_can_address_existing_xorbs_larger_than_64_mib() {
    let mut server = mockito::Server::new_async().await;
    let payload = encode_chunk(b"past the 64 MiB boundary", 0);
    let start = 64 * 1024 * 1024;
    let end = start + payload.len() as u64 - 1;
    let transfer = server
        .mock("GET", "/xorb")
        .with_status(206)
        .match_header("range", format!("bytes={start}-{end}").as_str())
        .with_header("content-range", &format!("bytes {start}-{end}/{}", end + 1))
        .with_body(payload)
        .create_async()
        .await;
    let client = XetClient::new(&server.url(), 2).unwrap();
    let fetch = Fetch {
        url: Url::parse(&format!("{}/xorb", server.url())).unwrap(),
        ranges: vec![FetchRange {
            chunks: Interval {
                start: 1024,
                end: 1025,
            },
            bytes: Interval { start, end },
        }],
    };
    assert_eq!(
        client.fetch(&fetch, &HashSet::from([1024])).await.unwrap()[&1024],
        b"past the 64 MiB boundary"
    );
    transfer.assert_async().await;
}

#[tokio::test]
async fn rejects_malformed_reconstructions_before_allocating_chunk_maps() {
    let server = mockito::Server::new_async().await;
    let client = XetClient::new(&server.url(), 2).unwrap();
    for (unpacked_length, end) in [(0, 65536), (65536, u64::MAX), (MAX_DECODED_BYTES + 1, 1)] {
        let reconstruction = Reconstruction {
            offset_into_first_range: 0,
            terms: vec![Term {
                hash: XORB_HASH.into(),
                unpacked_length,
                range: Interval { start: 0, end },
            }],
            xorbs: HashMap::new(),
            fetch_info: HashMap::new(),
        };
        assert!(client.assemble(reconstruction, 1).await.is_err());
    }
    assert!(Interval {
        start: 0,
        end: u64::MAX
    }
    .byte_length()
    .is_err());
    assert!(Interval { start: 2, end: 1 }.byte_length().is_err());
}

#[tokio::test]
async fn truncated_transfer_bodies_are_retried() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base_url = format!("http://{}", listener.local_addr().unwrap());
    let payload = encode_chunk(b"retry the interrupted response", 0);
    let size = payload.len();
    let server = tokio::spawn(async move {
        for attempt in 0..2 {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            while !request.ends_with(b"\r\n\r\n") {
                request.push(socket.read_u8().await.unwrap());
            }
            let headers = format!("HTTP/1.1 206 Partial Content\r\nContent-Length: {size}\r\nContent-Range: bytes 0-{}/{size}\r\nConnection: close\r\n\r\n", size - 1);
            socket.write_all(headers.as_bytes()).await.unwrap();
            let body = if attempt == 0 {
                &payload[..3]
            } else {
                &payload
            };
            socket.write_all(body).await.unwrap();
        }
    });
    let client = XetClient::new(&base_url, 2).unwrap();
    let fetch = Fetch {
        url: Url::parse(&format!("{base_url}/xorb")).unwrap(),
        ranges: vec![FetchRange {
            chunks: Interval { start: 0, end: 1 },
            bytes: Interval {
                start: 0,
                end: size as u64 - 1,
            },
        }],
    };
    let result = tokio::time::timeout(
        Duration::from_secs(5),
        client.fetch(&fetch, &HashSet::from([0])),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(result[&0], b"retry the interrupted response");
    server.await.unwrap();
}
