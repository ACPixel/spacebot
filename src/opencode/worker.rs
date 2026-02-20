//! OpenCode worker: drives an OpenCode session for coding tasks.
//!
//! Instead of running a Rig agent loop with shell/file/exec tools, this worker
//! delegates to an OpenCode subprocess that has its own codebase exploration,
//! context management, and tool suite. Communication happens over HTTP + SSE.

use crate::opencode::server::OpenCodeServerPool;
use crate::opencode::types::*;
use crate::{AgentId, ChannelId, ProcessEvent, ProcessId, WorkerId};

use anyhow::{Context as _, bail};
use futures::StreamExt as _;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, broadcast, mpsc};
use uuid::Uuid;

/// How long an interactive OpenCode worker waits for follow-up before finishing.
const INTERACTIVE_IDLE_TIMEOUT: Duration = Duration::from_secs(120);

/// An OpenCode-backed worker that drives a coding session via subprocess.
pub struct OpenCodeWorker {
    pub id: WorkerId,
    pub channel_id: Option<ChannelId>,
    pub agent_id: AgentId,
    pub task: String,
    pub directory: PathBuf,
    pub server_pool: Arc<OpenCodeServerPool>,
    pub event_tx: broadcast::Sender<ProcessEvent>,
    /// Input channel for interactive follow-ups (permissions, questions, user messages).
    pub input_rx: Option<mpsc::Receiver<String>>,
    /// System prompt injected into each OpenCode prompt.
    pub system_prompt: Option<String>,
    /// Model override (provider/model format like "anthropic/claude-sonnet-4-20250514").
    pub model: Option<String>,
}

/// Result of an OpenCode worker run.
pub struct OpenCodeWorkerResult {
    pub session_id: String,
    pub result_text: String,
}

impl OpenCodeWorker {
    /// Create a new OpenCode worker.
    pub fn new(
        channel_id: Option<ChannelId>,
        agent_id: AgentId,
        task: impl Into<String>,
        directory: PathBuf,
        server_pool: Arc<OpenCodeServerPool>,
        event_tx: broadcast::Sender<ProcessEvent>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            channel_id,
            agent_id,
            task: task.into(),
            directory,
            server_pool,
            event_tx,
            input_rx: None,
            system_prompt: None,
            model: None,
        }
    }

    /// Create an interactive OpenCode worker that accepts follow-up messages.
    pub fn new_interactive(
        channel_id: Option<ChannelId>,
        agent_id: AgentId,
        task: impl Into<String>,
        directory: PathBuf,
        server_pool: Arc<OpenCodeServerPool>,
        event_tx: broadcast::Sender<ProcessEvent>,
    ) -> (Self, mpsc::Sender<String>) {
        let (input_tx, input_rx) = mpsc::channel(32);
        let mut worker = Self::new(channel_id, agent_id, task, directory, server_pool, event_tx);
        worker.input_rx = Some(input_rx);
        (worker, input_tx)
    }

    /// Set the system prompt injected into OpenCode prompts.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set the model to use for this worker.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Run the worker: spawn/reuse an OpenCode server, create a session,
    /// send the task, monitor via SSE, and return the result.
    pub async fn run(mut self) -> anyhow::Result<OpenCodeWorkerResult> {
        self.send_status("starting OpenCode server");

        // Get or create server for this directory
        let server = self
            .server_pool
            .get_or_create(&self.directory)
            .await
            .with_context(|| {
                format!(
                    "failed to get OpenCode server for '{}'",
                    self.directory.display()
                )
            })?;

        self.send_status("creating session");

        // Create a session
        let session = {
            let guard = server.lock().await;
            guard
                .create_session(Some(format!("spacebot-worker-{}", self.id)))
                .await?
        };
        let mut session_id = session.id.clone();

        tracing::info!(
            worker_id = %self.id,
            session_id = %session_id,
            directory = %self.directory.display(),
            "OpenCode session created"
        );
        let mut active_model_override = self.model.clone();

        // Send initial task and process until completion.
        self.send_status("sending task to OpenCode");
        let mut latest_result_text = match self
            .run_prompt_once(
                &server,
                &session_id,
                &self.task,
                active_model_override.as_deref(),
            )
            .await
        {
            Ok(text) => text,
            Err(error)
                if active_model_override.is_some()
                    && is_model_not_found_error(&error.to_string()) =>
            {
                let requested_model = active_model_override.take().unwrap_or_default();
                let previous_session_id = session_id.clone();

                tracing::warn!(
                    worker_id = %self.id,
                    session_id = %previous_session_id,
                    requested_model,
                    %error,
                    "OpenCode model override unavailable, retrying in a fresh session with OpenCode default model"
                );

                self.send_status("configured model unavailable, retrying with OpenCode default");

                let replacement_session = {
                    let guard = server.lock().await;

                    if let Err(abort_error) = guard.abort_session(&previous_session_id).await {
                        tracing::debug!(
                            worker_id = %self.id,
                            session_id = %previous_session_id,
                            %abort_error,
                            "failed to abort old OpenCode session during model fallback"
                        );
                    }

                    guard
                        .create_session(Some(format!("spacebot-worker-{}", self.id)))
                        .await?
                };

                session_id = replacement_session.id;

                tracing::info!(
                    worker_id = %self.id,
                    previous_session_id = %previous_session_id,
                    new_session_id = %session_id,
                    "OpenCode session recreated after model override rejection"
                );

                self.run_prompt_once(&server, &session_id, &self.task, None)
                    .await?
            }
            Err(error) => return Err(error),
        };

        // Interactive follow-up loop
        if let Some(mut input_rx) = self.input_rx.take() {
            self.emit_worker_output(&latest_result_text);
            self.send_status("waiting for follow-up");

            loop {
                let follow_up = match tokio::time::timeout(INTERACTIVE_IDLE_TIMEOUT, input_rx.recv()).await {
                    Ok(Some(message)) => message,
                    Ok(None) => break,
                    Err(_) => {
                        self.send_status("no follow-up received, finishing");
                        break;
                    }
                };

                self.send_status("processing follow-up");
                match self
                    .run_prompt_once(
                        &server,
                        &session_id,
                        &follow_up,
                        active_model_override.as_deref(),
                    )
                    .await
                {
                    Ok(follow_up_result) => {
                        latest_result_text = follow_up_result;
                        self.emit_worker_output(&latest_result_text);
                        self.send_status("waiting for follow-up");
                    }
                    Err(error) => {
                        tracing::error!(
                            worker_id = %self.id,
                            %error,
                            "OpenCode follow-up failed"
                        );
                        self.send_status("failed");
                        break;
                    }
                }
            }
        }

        self.send_status("completed");

        tracing::info!(
            worker_id = %self.id,
            session_id = %session_id,
            "OpenCode worker completed"
        );

        Ok(OpenCodeWorkerResult {
            session_id,
            result_text: latest_result_text,
        })
    }

    /// Send one prompt and process SSE events until completion or error.
    async fn run_prompt_once(
        &self,
        server: &Arc<Mutex<crate::opencode::server::OpenCodeServer>>,
        session_id: &str,
        prompt_text: &str,
        model_override: Option<&str>,
    ) -> anyhow::Result<String> {
        // Subscribe before sending so we don't miss early events.
        let event_response = {
            let guard = server.lock().await;
            guard.subscribe_events().await?
        };

        let prompt_request = SendPromptRequest {
            parts: vec![PartInput::Text {
                text: prompt_text.to_string(),
                synthetic: None,
            }],
            system: self.system_prompt.clone(),
            model: model_override.and_then(parse_model_param),
            agent: None,
        };

        {
            let guard = server.lock().await;
            guard.send_prompt_async(session_id, &prompt_request).await?;
        }

        self.process_events(event_response, session_id, server).await
    }

    /// Process SSE events from the OpenCode event stream until the session
    /// goes idle or encounters an error.
    async fn process_events(
        &self,
        response: reqwest::Response,
        session_id: &str,
        server: &Arc<Mutex<crate::opencode::server::OpenCodeServer>>,
    ) -> anyhow::Result<String> {
        let mut stream = response.bytes_stream();
        let mut buffer = String::new();
        let mut last_text = String::new();
        let mut status_tracker = StatusTracker::new();
        // Guards: don't treat session.idle as completion until we've seen real work
        let mut has_received_event = false;
        let mut has_assistant_message = false;

        loop {
            let chunk = tokio::select! {
                chunk = stream.next() => chunk,
                _ = tokio::time::sleep(std::time::Duration::from_secs(600)) => {
                    bail!("OpenCode session timed out after 10 minutes of inactivity");
                }
            };

            let Some(chunk) = chunk else {
                // Stream ended -- if we have results, return them
                if has_assistant_message && !last_text.is_empty() {
                    return Ok(last_text);
                }
                bail!("OpenCode event stream ended before session completed");
            };

            let bytes = chunk.context("failed to read SSE chunk")?;
            buffer.push_str(&String::from_utf8_lossy(&bytes));

            // Parse SSE lines from buffer
            while let Some(event) = extract_sse_event(&mut buffer) {
                match self
                    .handle_sse_event(
                        &event,
                        session_id,
                        server,
                        &mut last_text,
                        &mut status_tracker,
                        &mut has_received_event,
                        &mut has_assistant_message,
                    )
                    .await
                {
                    EventAction::Continue => {}
                    EventAction::Complete => {
                        if !last_text.trim().is_empty() {
                            return Ok(last_text.clone());
                        }

                        if let Some(fallback_text) =
                            self.fetch_last_assistant_text(server, session_id).await
                        {
                            return Ok(fallback_text);
                        }

                        return Ok(last_text.clone());
                    }
                    EventAction::Error(message) => bail!("OpenCode session error: {message}"),
                }
            }
        }
    }

    /// Fetch the latest assistant text from session history when SSE text parts
    /// were not observed (rare but possible with provider/tool-only responses).
    async fn fetch_last_assistant_text(
        &self,
        server: &Arc<Mutex<crate::opencode::server::OpenCodeServer>>,
        session_id: &str,
    ) -> Option<String> {
        let guard = server.lock().await;
        let messages = guard.get_messages(session_id).await.ok()?;

        for message in messages.iter().rev() {
            let Some(parts) = message.get("parts").and_then(|parts| parts.as_array()) else {
                continue;
            };

            for part in parts {
                let is_text = part
                    .get("type")
                    .and_then(|value| value.as_str())
                    .map(|value| value == "text")
                    .unwrap_or(false);
                if !is_text {
                    continue;
                }

                let text = part
                    .get("text")
                    .and_then(|value| value.as_str())
                    .unwrap_or("")
                    .trim();
                if !text.is_empty() {
                    return Some(text.to_string());
                }
            }
        }

        None
    }

    /// Handle a single SSE event. Returns whether to continue, complete, or error.
    async fn handle_sse_event(
        &self,
        event: &SseEvent,
        session_id: &str,
        server: &Arc<Mutex<crate::opencode::server::OpenCodeServer>>,
        last_text: &mut String,
        status_tracker: &mut StatusTracker,
        has_received_event: &mut bool,
        has_assistant_message: &mut bool,
    ) -> EventAction {
        match event {
            SseEvent::MessageUpdated { info } => {
                *has_received_event = true;
                // Track assistant messages for idle guard
                if let Some(msg) = info {
                    if msg.role == "assistant" {
                        if let Some(sid) = &msg.session_id {
                            if sid == session_id {
                                *has_assistant_message = true;
                            }
                        }
                    }
                }
                EventAction::Continue
            }

            SseEvent::MessagePartUpdated { part, .. } => {
                *has_received_event = true;
                match part {
                    Part::Text {
                        text,
                        session_id: part_session,
                        ..
                    } => {
                        if let Some(sid) = part_session {
                            if sid != session_id {
                                return EventAction::Continue;
                            }
                        }
                        *has_assistant_message = true;
                        *last_text = text.clone();
                    }
                    Part::Tool {
                        id: part_id,
                        tool,
                        state,
                        call_id,
                        session_id: part_session,
                        ..
                    } => {
                        if let Some(sid) = part_session {
                            if sid != session_id {
                                return EventAction::Continue;
                            }
                        }
                        *has_assistant_message = true;
                        if let Some(tool_state) = state {
                            let tool_name = tool.as_deref().unwrap_or("tool");
                            let call_key = tool_call_key(part_id, call_id.as_deref(), tool_name);

                            match tool_state {
                                ToolState::Running { title, input, .. } => {
                                    if status_tracker.mark_running(&call_key) {
                                        self.send_tool_started(tool_name);
                                    }
                                    let detail =
                                        tool_status_detail(title.as_deref(), input.as_ref());
                                    let status =
                                        format_tool_status("running", tool_name, detail.as_deref());
                                    self.send_status_tracked(status_tracker, &status);
                                }
                                ToolState::Completed { title, input, .. } => {
                                    if status_tracker
                                        .mark_terminal(&call_key, ToolLifecycle::Completed)
                                    {
                                        self.send_tool_completed(tool_name);
                                    }
                                    let detail =
                                        tool_status_detail(title.as_deref(), input.as_ref());
                                    let status = format_tool_status(
                                        "completed",
                                        tool_name,
                                        detail.as_deref(),
                                    );
                                    self.send_status_tracked(status_tracker, &status);
                                }
                                ToolState::Error { error, input, .. } => {
                                    if status_tracker.mark_terminal(&call_key, ToolLifecycle::Error)
                                    {
                                        self.send_tool_completed(tool_name);
                                    }
                                    let detail = error
                                        .as_deref()
                                        .map(clean_status_text)
                                        .filter(|s| !s.is_empty())
                                        .or_else(|| tool_status_detail(None, input.as_ref()));
                                    let status =
                                        format_tool_status("error", tool_name, detail.as_deref());
                                    self.send_status_tracked(status_tracker, &status);
                                }
                                ToolState::Pending { .. } => {
                                    status_tracker.mark_pending(&call_key);
                                }
                            }
                        }
                    }
                    _ => {}
                }
                EventAction::Continue
            }

            SseEvent::SessionIdle {
                session_id: event_session_id,
            } => {
                if event_session_id != session_id {
                    return EventAction::Continue;
                }

                // Guard: don't complete until we've seen actual work.
                // OpenCode can send an early idle event before the prompt is processed.
                if !*has_received_event || !*has_assistant_message {
                    tracing::trace!(
                        worker_id = %self.id,
                        has_received_event,
                        has_assistant_message,
                        "ignoring early session.idle"
                    );
                    return EventAction::Continue;
                }

                EventAction::Complete
            }

            SseEvent::SessionError {
                session_id: event_session_id,
                error,
            } => {
                if event_session_id.as_deref() != Some(session_id) {
                    return EventAction::Continue;
                }

                let message = extract_session_error_message(error.as_ref())
                    .unwrap_or_else(|| "unknown error".to_string());

                if message == "unknown error" {
                    tracing::warn!(
                        worker_id = %self.id,
                        session_id,
                        raw_error = ?error,
                        "OpenCode session.error without readable message"
                    );
                }

                self.send_status_tracked(status_tracker, &format!("session error: {message}"));
                EventAction::Error(message)
            }

            SseEvent::PermissionRequested(permission) => {
                if permission.session_id != session_id {
                    return EventAction::Continue;
                }

                let patterns = permission.pattern_list();
                let description = permission.summary();

                tracing::info!(
                    worker_id = %self.id,
                    permission_id = %permission.id,
                    permission_type = %permission.kind(),
                    patterns = ?patterns,
                    "OpenCode requesting permission"
                );

                let status = format!("permission requested: {description}");
                self.send_status_tracked(status_tracker, &status);

                let _ = self.event_tx.send(ProcessEvent::WorkerPermission {
                    agent_id: self.agent_id.clone(),
                    worker_id: self.id,
                    channel_id: self.channel_id.clone(),
                    permission_id: permission.id.clone(),
                    description: description.clone(),
                    patterns,
                });

                // Auto-allow (OPENCODE_CONFIG_CONTENT should prevent most prompts)
                let guard = server.lock().await;
                if let Err(error) = guard
                    .reply_permission(
                        &permission.session_id,
                        &permission.id,
                        PermissionReply::Once,
                    )
                    .await
                {
                    tracing::warn!(
                        worker_id = %self.id,
                        permission_id = %permission.id,
                        %error,
                        "failed to auto-reply permission"
                    );
                    self.send_status_tracked(
                        status_tracker,
                        &format!("permission reply failed: {description}"),
                    );
                } else {
                    self.send_status_tracked(
                        status_tracker,
                        &format!("permission approved: {description}"),
                    );
                }

                EventAction::Continue
            }

            SseEvent::QuestionAsked(question) => {
                if question.session_id != session_id {
                    return EventAction::Continue;
                }

                tracing::info!(
                    worker_id = %self.id,
                    question_id = %question.id,
                    question_count = question.questions.len(),
                    "OpenCode asking question"
                );

                let question_summary = summarize_questions(&question.questions);
                self.send_status_tracked(
                    status_tracker,
                    &format!("question asked: {question_summary}"),
                );

                let _ = self.event_tx.send(ProcessEvent::WorkerQuestion {
                    agent_id: self.agent_id.clone(),
                    worker_id: self.id,
                    channel_id: self.channel_id.clone(),
                    question_id: question.id.clone(),
                    questions: question
                        .questions
                        .iter()
                        .map(|q| QuestionInfo {
                            question: q.question.clone(),
                            header: q.header.clone(),
                            options: q.options.clone(),
                        })
                        .collect(),
                });

                // Auto-select first option
                let answers: Vec<QuestionAnswer> = question
                    .questions
                    .iter()
                    .map(|q| {
                        if let Some(first_option) = q.options.first() {
                            QuestionAnswer {
                                label: first_option.label.clone(),
                                description: first_option.description.clone(),
                            }
                        } else {
                            QuestionAnswer {
                                label: "continue".to_string(),
                                description: None,
                            }
                        }
                    })
                    .collect();

                let guard = server.lock().await;
                if let Err(error) = guard.reply_question(&question.id, answers).await {
                    tracing::warn!(
                        worker_id = %self.id,
                        question_id = %question.id,
                        %error,
                        "failed to auto-reply question"
                    );
                    self.send_status_tracked(
                        status_tracker,
                        &format!("failed to answer question: {question_summary}"),
                    );
                } else {
                    self.send_status_tracked(
                        status_tracker,
                        &format!("question answered: {question_summary}"),
                    );
                }

                EventAction::Continue
            }

            SseEvent::SessionStatus {
                session_id: event_session_id,
                status,
            } => {
                if event_session_id != session_id {
                    return EventAction::Continue;
                }
                match status {
                    SessionStatusPayload::Retry {
                        attempt, message, ..
                    } => {
                        let description = message.as_deref().unwrap_or("rate limited");
                        self.send_status_tracked(
                            status_tracker,
                            &format!("retry attempt {attempt}: {description}"),
                        );
                    }
                    SessionStatusPayload::Busy => {
                        if !status_tracker.has_running_tools() {
                            self.send_status_tracked(status_tracker, "waiting for model response");
                        }
                    }
                    SessionStatusPayload::Idle => {}
                }
                EventAction::Continue
            }

            _ => EventAction::Continue,
        }
    }

    /// Send a status update via the process event bus.
    fn send_status(&self, status: &str) {
        let status = cap_status(status);
        let _ = self.event_tx.send(ProcessEvent::WorkerStatus {
            agent_id: self.agent_id.clone(),
            worker_id: self.id,
            channel_id: self.channel_id.clone(),
            status,
        });
    }

    /// Send a status update with de-duplication and repeat throttling.
    fn send_status_tracked(&self, status_tracker: &mut StatusTracker, status: &str) {
        let status = cap_status(status);
        if !status_tracker.should_emit_status(&status) {
            return;
        }

        self.send_status(&status);
        status_tracker.record_status(status);
    }

    /// Emit a synthetic tool-start event so OpenCode workers show rich live progress.
    fn send_tool_started(&self, tool_name: &str) {
        let _ = self.event_tx.send(ProcessEvent::ToolStarted {
            agent_id: self.agent_id.clone(),
            process_id: ProcessId::Worker(self.id),
            channel_id: self.channel_id.clone(),
            tool_name: tool_name.to_string(),
        });
    }

    /// Emit a synthetic tool-complete event so tool call counts increment.
    fn send_tool_completed(&self, tool_name: &str) {
        let _ = self.event_tx.send(ProcessEvent::ToolCompleted {
            agent_id: self.agent_id.clone(),
            process_id: ProcessId::Worker(self.id),
            channel_id: self.channel_id.clone(),
            tool_name: tool_name.to_string(),
            result: String::new(),
        });
    }

    /// Emit non-terminal worker output for interactive sessions.
    fn emit_worker_output(&self, output: &str) {
        if output.trim().is_empty() {
            return;
        }

        let _ = self.event_tx.send(ProcessEvent::WorkerOutput {
            agent_id: self.agent_id.clone(),
            worker_id: self.id,
            channel_id: self.channel_id.clone(),
            output: cap_status(output),
        });
    }
}

/// Result of processing a single SSE event.
enum EventAction {
    Continue,
    Complete,
    Error(String),
}

/// Per-prompt status tracker used to de-duplicate noisy SSE updates.
#[derive(Debug, Default)]
struct StatusTracker {
    tool_states: HashMap<String, ToolLifecycle>,
    last_status: Option<String>,
    last_status_at: Option<Instant>,
}

impl StatusTracker {
    fn new() -> Self {
        Self::default()
    }

    fn mark_pending(&mut self, call_key: &str) {
        self.tool_states
            .entry(call_key.to_string())
            .or_insert(ToolLifecycle::Pending);
    }

    /// Mark a tool call as running.
    /// Returns true if this is the first transition into running.
    fn mark_running(&mut self, call_key: &str) -> bool {
        match self.tool_states.get(call_key).copied() {
            Some(ToolLifecycle::Running) => false,
            Some(ToolLifecycle::Completed) | Some(ToolLifecycle::Error) => false,
            Some(ToolLifecycle::Pending) | None => {
                self.tool_states
                    .insert(call_key.to_string(), ToolLifecycle::Running);
                true
            }
        }
    }

    /// Mark a tool call as terminal (completed or error).
    /// Returns true if this is the first transition into a terminal state.
    fn mark_terminal(&mut self, call_key: &str, terminal: ToolLifecycle) -> bool {
        if !matches!(terminal, ToolLifecycle::Completed | ToolLifecycle::Error) {
            return false;
        }

        let already_terminal = matches!(
            self.tool_states.get(call_key),
            Some(ToolLifecycle::Completed) | Some(ToolLifecycle::Error)
        );

        self.tool_states.insert(call_key.to_string(), terminal);
        !already_terminal
    }

    fn has_running_tools(&self) -> bool {
        self.tool_states
            .values()
            .any(|state| *state == ToolLifecycle::Running)
    }

    fn should_emit_status(&self, status: &str) -> bool {
        const REPEAT_INTERVAL: Duration = Duration::from_secs(8);

        if self.last_status.as_deref() != Some(status) {
            return true;
        }

        self.last_status_at
            .map(|when| when.elapsed() >= REPEAT_INTERVAL)
            .unwrap_or(true)
    }

    fn record_status(&mut self, status: String) {
        self.last_status = Some(status);
        self.last_status_at = Some(Instant::now());
    }
}

/// Tool lifecycle states seen in SSE events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolLifecycle {
    Pending,
    Running,
    Completed,
    Error,
}

fn tool_call_key(part_id: &str, call_id: Option<&str>, tool_name: &str) -> String {
    if let Some(call_id) = call_id {
        format!("{tool_name}:{call_id}")
    } else {
        format!("{tool_name}:{part_id}")
    }
}

fn tool_status_detail(title: Option<&str>, input: Option<&serde_json::Value>) -> Option<String> {
    title
        .map(clean_status_text)
        .filter(|text| !text.is_empty())
        .or_else(|| summarize_tool_input(input))
}

fn summarize_tool_input(input: Option<&serde_json::Value>) -> Option<String> {
    let value = input?;

    if let Some(text) = value.as_str() {
        let cleaned = clean_status_text(text);
        return (!cleaned.is_empty()).then_some(cleaned);
    }

    let Some(object) = value.as_object() else {
        return None;
    };

    for key in [
        "description",
        "command",
        "filePath",
        "path",
        "pattern",
        "query",
        "url",
        "task",
        "header",
        "question",
    ] {
        if let Some(text) = object.get(key).and_then(|v| v.as_str()) {
            let cleaned = clean_status_text(text);
            if !cleaned.is_empty() {
                return Some(cleaned);
            }
        }
    }

    if let Some(paths) = object.get("paths").and_then(|v| v.as_array()) {
        if let Some(first_path) = paths.first().and_then(|v| v.as_str()) {
            let cleaned = clean_status_text(first_path);
            if !cleaned.is_empty() {
                return Some(cleaned);
            }
        }
    }

    None
}

fn summarize_questions(questions: &[QuestionInfo]) -> String {
    if let Some(question) = questions.first() {
        if let Some(text) = question
            .question
            .as_deref()
            .filter(|text| !text.trim().is_empty())
        {
            return clean_status_text(text);
        }

        if let Some(header) = question
            .header
            .as_deref()
            .filter(|text| !text.trim().is_empty())
        {
            return clean_status_text(header);
        }
    }

    format!("{} question(s)", questions.len())
}

fn format_tool_status(prefix: &str, tool_name: &str, detail: Option<&str>) -> String {
    match detail {
        Some(detail) if !detail.is_empty() => format!("{prefix} {tool_name}: {detail}"),
        _ => format!("{prefix} {tool_name}"),
    }
}

fn clean_status_text(text: &str) -> String {
    let compact = text.split_whitespace().collect::<Vec<_>>().join(" ");
    truncate_text(&compact, 160)
}

fn extract_session_error_message(error: Option<&serde_json::Value>) -> Option<String> {
    let error = error?;
    extract_error_value_message(error)
}

fn extract_error_value_message(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(text) => {
            let cleaned = clean_status_text(text);
            (!cleaned.is_empty()).then_some(cleaned)
        }
        serde_json::Value::Object(object) => {
            for key in ["message", "error", "detail", "description", "reason"] {
                if let Some(candidate) = object.get(key) {
                    if let Some(message) = extract_error_value_message(candidate) {
                        return Some(message);
                    }
                }
            }

            if let Some(code) = object.get("code").and_then(|value| value.as_str()) {
                let cleaned = clean_status_text(code);
                if !cleaned.is_empty() {
                    return Some(cleaned);
                }
            }

            let serialized = clean_status_text(&value.to_string());
            (!serialized.is_empty()).then_some(serialized)
        }
        serde_json::Value::Array(items) => {
            for item in items {
                if let Some(message) = extract_error_value_message(item) {
                    return Some(message);
                }
            }

            let serialized = clean_status_text(&value.to_string());
            (!serialized.is_empty()).then_some(serialized)
        }
        _ => {
            let serialized = clean_status_text(&value.to_string());
            (!serialized.is_empty()).then_some(serialized)
        }
    }
}

fn cap_status(status: &str) -> String {
    truncate_text(status, 256)
}

fn truncate_text(text: &str, max_chars: usize) -> String {
    if text.len() <= max_chars {
        return text.to_string();
    }

    let end = text.floor_char_boundary(max_chars);
    let boundary = text[..end].rfind(char::is_whitespace).unwrap_or(end);
    format!("{}...", text[..boundary].trim_end())
}

/// Parse an SSE event from a buffer. Parses the `{ type, properties }` envelope
/// and converts to our `SseEvent` enum. Returns None if no complete event is available.
fn extract_sse_event(buffer: &mut String) -> Option<SseEvent> {
    // SSE format: lines starting with "data: " followed by JSON, terminated by
    // a blank line. We may also see "event:" and "id:" lines which we ignore.
    loop {
        let (block_end, separator_len) = if let Some(pos) = buffer.find("\n\n") {
            (pos, 2)
        } else if let Some(pos) = buffer.find("\r\n\r\n") {
            (pos, 4)
        } else {
            return None;
        };
        let block = buffer[..block_end].to_string();
        *buffer = buffer[block_end + separator_len..].to_string();

        // Extract all data lines from the block
        let mut data_parts = Vec::new();
        for line in block.lines() {
            if let Some(data) = line.strip_prefix("data: ") {
                data_parts.push(data);
            } else if let Some(data) = line.strip_prefix("data:") {
                data_parts.push(data);
            }
        }

        if data_parts.is_empty() {
            continue;
        }

        let json_str = data_parts.join("\n");
        if json_str.is_empty() {
            continue;
        }

        // Parse the envelope first, then convert to our event type
        match serde_json::from_str::<SseEventEnvelope>(&json_str) {
            Ok(envelope) => return Some(SseEvent::from_envelope(envelope)),
            Err(error) => {
                tracing::trace!(
                    %error,
                    json = %json_str,
                    "failed to parse SSE event envelope, skipping"
                );
                continue;
            }
        }
    }
}

/// Parse a model string like "anthropic/claude-sonnet-4-20250514" into a ModelParam.
fn parse_model_param(model: &str) -> Option<ModelParam> {
    let (provider, model_id) = model.split_once('/')?;
    Some(ModelParam {
        provider_id: provider.to_string(),
        model_id: model_id.to_string(),
    })
}

fn is_model_not_found_error(message: &str) -> bool {
    let normalized = message.to_ascii_lowercase();
    normalized.contains("model not found")
        || normalized.contains("unknown model")
        || normalized.contains("invalid model")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_session_error_message_from_top_level_message() {
        let value = serde_json::json!({ "message": "provider request failed" });
        let message = extract_session_error_message(Some(&value));
        assert_eq!(message.as_deref(), Some("provider request failed"));
    }

    #[test]
    fn extracts_session_error_message_from_nested_error_string() {
        let value = serde_json::json!({ "error": "rate limited" });
        let message = extract_session_error_message(Some(&value));
        assert_eq!(message.as_deref(), Some("rate limited"));
    }

    #[test]
    fn extracts_session_error_message_from_array_payload() {
        let value = serde_json::json!({
            "detail": [
                { "message": "quota exceeded" },
                "fallback"
            ]
        });
        let message = extract_session_error_message(Some(&value));
        assert_eq!(message.as_deref(), Some("quota exceeded"));
    }

    #[test]
    fn detects_model_not_found_errors() {
        assert!(is_model_not_found_error(
            "OpenCode session error: Model not found: openrouter/anthropic/claude-haiku-4.5"
        ));
        assert!(is_model_not_found_error("unknown model selected"));
        assert!(!is_model_not_found_error("permission denied"));
    }
}
