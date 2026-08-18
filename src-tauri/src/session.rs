//! Agent sessions driven from the UI.
//!
//! Bridges the runner's `async` world to the frontend: run events are emitted
//! as Tauri events, and calls the rules escalated are held open until the user
//! answers.
//!
//! The approval bridge fails closed. If the window is closed, the frontend
//! reloads, or the sender is dropped for any other reason, the pending call
//! resolves as *rejected* rather than approved — a lost message must never
//! become permission.

use std::collections::HashMap;
use std::sync::Arc;
use tauri::{AppHandle, Emitter};
use tokio::sync::{oneshot, Mutex};

use crate::agent::approver::{AssistedGate, ModelOracle};
use crate::agent::runner::{ApprovalGate, ProposedCall};
use crate::ollama::OllamaClient;
use crate::perms::Outcome;

/// Calls waiting on a human answer, keyed by call id.
pub type PendingApprovals = Arc<Mutex<HashMap<String, oneshot::Sender<Outcome>>>>;

/// Asks the user, through the UI.
pub struct UiGate {
    app: AppHandle,
    pending: PendingApprovals,
}

impl UiGate {
    pub fn new(app: AppHandle, pending: PendingApprovals) -> Self {
        Self { app, pending }
    }
}

#[async_trait::async_trait]
impl ApprovalGate for UiGate {
    async fn decide(&self, call: &ProposedCall) -> Outcome {
        let (tx, rx) = oneshot::channel();
        self.pending.lock().await.insert(call.id.clone(), tx);

        if self.app.emit("agent://approval", call).is_err() {
            // No window to ask. Refusing is the only safe answer.
            self.pending.lock().await.remove(&call.id);
            return Outcome::UserRejected;
        }

        // A dropped sender means the UI went away mid-question.
        rx.await.unwrap_or(Outcome::UserRejected)
    }
}

/// Settle a call the UI was asked about.
///
/// Returns whether a waiting call was actually found — a stale id from a
/// previous run is reported rather than silently ignored.
pub async fn resolve(pending: &PendingApprovals, id: &str, outcome: Outcome) -> bool {
    match pending.lock().await.remove(id) {
        Some(tx) => tx.send(outcome).is_ok(),
        None => false,
    }
}

/// Drop every waiting call, refusing each. Used when a run is cancelled.
pub async fn reject_all(pending: &PendingApprovals) {
    let waiting: Vec<_> = pending.lock().await.drain().collect();
    for (_, tx) in waiting {
        let _ = tx.send(Outcome::UserRejected);
    }
}

/// Build the gate stack: the classifier first, the user behind it.
///
/// When `assisted` is false the classifier is bypassed entirely, for users who
/// want no model involvement in permission decisions.
pub fn build_gate(
    app: AppHandle,
    pending: PendingApprovals,
    client: OllamaClient,
    classifier_model: Option<String>,
    assisted: bool,
) -> Box<dyn ApprovalGate> {
    let ui = UiGate::new(app, pending);
    match (assisted, classifier_model) {
        (true, Some(model)) => Box::new(AssistedGate::new(ModelOracle::new(client, model), ui)),
        _ => Box::new(ui),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pending() -> PendingApprovals {
        Arc::new(Mutex::new(HashMap::new()))
    }

    #[tokio::test]
    async fn resolving_delivers_the_answer() {
        let p = pending();
        let (tx, rx) = oneshot::channel();
        p.lock().await.insert("abc".into(), tx);

        assert!(resolve(&p, "abc", Outcome::UserApproved).await);
        assert_eq!(rx.await.unwrap(), Outcome::UserApproved);
    }

    #[tokio::test]
    async fn resolving_an_unknown_id_reports_failure() {
        assert!(!resolve(&pending(), "nope", Outcome::UserApproved).await);
    }

    #[tokio::test]
    async fn a_call_cannot_be_resolved_twice() {
        let p = pending();
        let (tx, _rx) = oneshot::channel();
        p.lock().await.insert("abc".into(), tx);

        assert!(resolve(&p, "abc", Outcome::UserApproved).await);
        // A replayed approval must not settle anything a second time.
        assert!(!resolve(&p, "abc", Outcome::UserApproved).await);
    }

    #[tokio::test]
    async fn cancelling_refuses_every_waiting_call() {
        let p = pending();
        let (tx1, rx1) = oneshot::channel();
        let (tx2, rx2) = oneshot::channel();
        p.lock().await.insert("a".into(), tx1);
        p.lock().await.insert("b".into(), tx2);

        reject_all(&p).await;

        assert_eq!(rx1.await.unwrap(), Outcome::UserRejected);
        assert_eq!(rx2.await.unwrap(), Outcome::UserRejected);
        assert!(p.lock().await.is_empty());
    }

    #[tokio::test]
    async fn a_dropped_sender_reads_as_refusal() {
        // The UI going away mid-question must not approve anything.
        let (tx, rx) = oneshot::channel::<Outcome>();
        drop(tx);
        assert_eq!(rx.await.unwrap_or(Outcome::UserRejected), Outcome::UserRejected);
    }
}
