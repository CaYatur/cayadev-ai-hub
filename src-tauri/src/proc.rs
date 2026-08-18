//! Spawning without flashing a console window.
//!
//! A Windows GUI application has no console, so every child process the agent
//! starts gets one created for it — a black window that appears and vanishes.
//! With a tool call or two per step that is constant flickering, and it looks
//! like the app is doing something it should not be.
//!
//! `CREATE_NO_WINDOW` suppresses it. Every spawn in the app goes through here so
//! a new call site cannot forget.

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// An async command that will not open a console window.
pub fn command(program: &str) -> tokio::process::Command {
    let mut cmd = tokio::process::Command::new(program);
    // `tokio::process::Command` has its own `creation_flags` on Windows, so no
    // extension trait is needed here — unlike the std version below.
    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);
    cmd
}

/// A blocking command that will not open a console window.
///
/// Used for the availability probes that run before the async runtime is
/// necessarily involved.
pub fn sync_command(program: &str) -> std::process::Command {
    let mut cmd = std::process::Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}
