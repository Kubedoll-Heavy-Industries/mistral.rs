# API Layer Review

**Files**: `mistralrs-server-core/src/*.rs`, `mistralrs-pyo3/src/lib.rs`, `mistralrs/src/*.rs`, `mistralrs-server/src/interactive_mode.rs`
**Lines Changed**: ~1750

## Summary

The API layer has been updated to use the new request type system:
- `RequestMessage` enum replaced by `InferenceOperation` enum
- `VisionChat` and `Chat` unified into single variant with `attachments`
- New `ThinkingMode` enum consolidates thinking options
- `RequestLike` trait replaced by concrete `Chat` struct with `From` conversions
- `NormalRequest` now wraps `InferenceInput { op: InferenceOperation, exec: InferenceExec }`

## Strengths

### 1. Cleaner Type Hierarchy
Clear separation:
- **Transport**: `id`, `response`, `model_id` (on `InferenceRequest`)
- **Execution**: `is_streaming`, `truncate_sequence` (in `InferenceExec`)
- **Operation**: operation-specific fields (in `InferenceOperation`)

### 2. Unified Attachments Model
Eliminates awkward `VisionChat` vs `Chat` dichotomy:
```rust
pub enum ChatAttachment {
    Image(image::DynamicImage),
    Audio(AudioInput),
}
```

### 3. ThinkingMode Consolidation
```rust
pub enum ThinkingMode {
    Bool(bool),
    Effort(ReasoningEffort),
}
```

### 4. Simpler Model API
```rust
// Old:
pub async fn send_chat_request<R: RequestLike>(&self, mut request: R)

// New:
pub async fn send_chat_request(&self, request: Chat)
```

### 5. Good Backward Compatibility
`From` implementations preserve ergonomics:
```rust
impl From<TextMessages> for Chat { ... }
impl From<VisionMessages> for Chat { ... }
```

## Issues

### Issue 1: ⚠️ `adapters` Field Silently Dropped - PRE-EXISTING
**Location**: `messages.rs:627-628` and `messages.rs:741-779`
**Severity**: Low (pre-existing dead code, not a regression)

`RequestBuilder` has adapters field and setter:
```rust
pub fn set_adapters(mut self, adapters: Vec<String>) -> Self {
    self.adapters = adapters;
    self
}
```

But `From<RequestBuilder> for Chat` ignores it.

**Investigation Finding**: The `take_adapters()` method was defined in the `RequestLike` trait but **never called anywhere in the entire codebase**. The adapters field was dead code before this refactoring - users could set adapters but they were never used.

**Impact**: None (no regression - feature was never working)

**Recommendation**:
1. Remove the dead `adapters` field and `set_adapters()` method from `RequestBuilder`
2. Or implement adapter selection properly in a future PR

### Issue 2: CRITICAL - Massive Code Duplication (Python)
**Location**: `pyo3/lib.rs` (multiple locations)
**Severity**: Critical

`SamplingParams` and `InferenceOperation::Chat` construction duplicated 6+ times:
```rust
// This block copy-pasted 6 times with minor variations:
InferenceOperation::Chat {
    messages: messages_vec,
    attachments,
    thinking,
    sampling_params: SamplingParams {
        temperature: request.temperature,
        top_k: request.top_k,
        top_p: request.top_p,
        // ... 10 more fields
    },
    return_logprobs: request.logprobs,
    constraint: constraint.clone(),
}
```

**Recommendation**: Extract helper:
```rust
fn build_chat_operation(
    messages: Vec<...>,
    attachments: Vec<ChatAttachment>,
    request: &ChatRequest,
    constraint: Constraint,
    tools: Option<Vec<Tool>>,
) -> InferenceOperation
```

### Issue 3: HIGH - ThinkingMode Conversion Duplicated
**Location**: Multiple files (6+ times)
**Severity**: High

Same logic repeated verbatim:
```rust
let thinking = match (request.enable_thinking, reasoning_effort) {
    (_, Some(effort)) => Some(mistralrs_core::ThinkingMode::Effort(effort)),
    (Some(b), None) => Some(mistralrs_core::ThinkingMode::Bool(b)),
    (None, None) => None,
};
```

**Recommendation**: Add conversion method:
```rust
impl ThinkingMode {
    pub fn from_options(
        enable_thinking: Option<bool>,
        reasoning_effort: Option<ReasoningEffort>,
    ) -> Option<Self> {
        match (enable_thinking, reasoning_effort) {
            (_, Some(effort)) => Some(Self::Effort(effort)),
            (Some(b), None) => Some(Self::Bool(b)),
            (None, None) => None,
        }
    }
}
```

### Issue 4: HIGH - Attachment Order Lost (Python)
**Location**: `pyo3/lib.rs:1097-1109`
**Severity**: High

```rust
// All images before all audios - loses interleaved order
for url in image_urls {
    attachments.push(ChatAttachment::Image(image));
}
for url in audio_urls {
    attachments.push(ChatAttachment::Audio(audio));
}
```

Server-core preserves order by processing in encounter order.

**Recommendation**: Process in encounter order:
```rust
for (url, kind) in attachment_urls {
    match kind {
        AttachmentKind::Image => attachments.push(ChatAttachment::Image(...)),
        AttachmentKind::Audio => attachments.push(ChatAttachment::Audio(...)),
    }
}
```

### Issue 5: MEDIUM - Local Shadowing Enum
**Location**: `chat_completion.rs:43-49`
**Severity**: Medium

```rust
enum RequestMessage {
    Chat {
        messages: Vec<IndexMap<String, MessageContent>>,
        attachments: Vec<ChatAttachment>,
        thinking: Option<ThinkingMode>,
    },
}
```

Single-variant enum used only for pattern matching before constructing final type.

**Recommendation**: Remove, construct `InferenceOperation::Chat` directly.

### Issue 6: MEDIUM - Request ID Inconsistency
**Location**: Multiple files
**Severity**: Medium

- `model.rs`: `uuid::Uuid::nil()`
- `pyo3/lib.rs`: `uuid::Uuid::now_v7()`
- `server-core`: `state.next_request_id()`

**Recommendation**: Document ID generation strategy.

### Issue 7: LOW - Formatting Issues
**Location**: `chat_completion.rs:435-448`
**Severity**: Low

Inconsistent indentation in nested if-matches. Run `cargo fmt`.

## Breaking Changes Assessment

| Change | Impact | Mitigation |
|--------|--------|------------|
| `RequestLike` trait removed | High for SDK users | `From<TextMessages>` preserves most patterns |
| `send_chat_request<R>` now takes `Chat` | Medium | Users add `.into()` |
| `adapters` field ignored | **High for LoRA users** | **Needs fix or warning** |
| `VisionChat` merged into `Chat` | Low | Semantic equivalence |

## Recommendations Summary

1. **Add adapters back** or deprecate setter with warning
2. **Extract helpers** in pyo3 to reduce duplication
3. **Add `ThinkingMode::from_options()`** helper
4. **Fix attachment order** in Python bindings
5. **Remove local shadowing enum**
6. **Run `cargo fmt`**

## Verdict

The API migration is **largely successful** with good backward compatibility. However:

1. **Critical**: `adapters` field silently dropped breaks LoRA users
2. **High**: Massive code duplication in pyo3 needs cleanup
3. **High**: Attachment order lost in Python bindings

The adapters issue is the only true blocker; duplication is technical debt for post-merge.
