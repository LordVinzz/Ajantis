import { useSyncExternalStore } from "react";

import type { StreamEvent } from "./types";

interface StreamSnapshot {
  version: number;
  events: StreamEvent[];
}

class StreamStore {
  private snapshot: StreamSnapshot = { version: 0, events: [] };
  private listeners = new Set<() => void>();
  private notifyFrame: number | null = null;

  private scheduleNotify() {
    if (this.notifyFrame != null) {
      return;
    }
    this.notifyFrame = window.requestAnimationFrame(() => {
      this.notifyFrame = null;
      this.listeners.forEach((listener) => listener());
    });
  }

  subscribe = (listener: () => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  getSnapshot = () => this.snapshot;

  publish = (event: StreamEvent) => {
    this.snapshot = {
      version: this.snapshot.version + 1,
      events: [...this.snapshot.events, event],
    };
    this.scheduleNotify();
  };

  clear = () => {
    if (this.snapshot.events.length === 0) {
      return;
    }
    this.snapshot = { version: this.snapshot.version + 1, events: [] };
    this.scheduleNotify();
  };
}

export const streamStore = new StreamStore();

export function useStreamSnapshot() {
  return useSyncExternalStore(streamStore.subscribe, streamStore.getSnapshot);
}
