use std::sync::Arc;

use tauri::ipc::Channel;

use crate::chat::StreamEvent;

pub trait EventSink: Send + Sync {
    fn send(&self, event: StreamEvent) -> Result<(), String>;
}

pub type SharedEventSink = Arc<dyn EventSink>;

struct ChannelSink {
    channel: Channel<StreamEvent>,
}

impl EventSink for ChannelSink {
    fn send(&self, event: StreamEvent) -> Result<(), String> {
        self.channel.send(event).map_err(|e| e.to_string())
    }
}

pub fn channel_event_sink(channel: Channel<StreamEvent>) -> SharedEventSink {
    Arc::new(ChannelSink { channel })
}

struct CallbackSink {
    callback: Arc<dyn Fn(StreamEvent) -> Result<(), String> + Send + Sync>,
}

impl EventSink for CallbackSink {
    fn send(&self, event: StreamEvent) -> Result<(), String> {
        (self.callback)(event)
    }
}

pub fn callback_event_sink<F>(callback: F) -> SharedEventSink
where
    F: Fn(StreamEvent) -> Result<(), String> + Send + Sync + 'static,
{
    Arc::new(CallbackSink {
        callback: Arc::new(callback),
    })
}
