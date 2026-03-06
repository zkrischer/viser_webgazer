import React from "react";
import { ViewerContext } from "./ViewerContext";

// Optional type for globally loaded webgazer.
type WebGazerLike = {
  setGazeListener: (
    cb: (data: { x: number; y: number } | null, elapsedTime: number) => void,
  ) => WebGazerLike;
  begin: () => Promise<unknown> | unknown;
  end?: () => void;
};

/**
 * Streams normalized gaze samples to the Python server.
 *
 * If `window.webgazer` is available, this component uses it directly.
 * Otherwise, it falls back to pointer position as a synthetic gaze signal so
 * backend message plumbing can still be tested end-to-end.
 */
export function GazeTracker() {
  const viewer = React.useContext(ViewerContext)!;
  const viewerMutable = viewer.mutable.current;

  React.useEffect(() => {
    if (viewer.messageSource !== "websocket") return;

    const maxHz = 20.0;
    const minDtSec = 1.0 / maxHz;
    let lastSentSec = 0.0;

    const sendNormalized = (xNorm: number, yNorm: number, timestampSec: number) => {
      if (!Number.isFinite(xNorm) || !Number.isFinite(yNorm)) return;
      if (!Number.isFinite(timestampSec)) return;
      if (timestampSec - lastSentSec < minDtSec) return;

      lastSentSec = timestampSec;

      viewerMutable.sendMessage({
        type: "ViewerGazeMessage",
        x: Math.max(0.0, Math.min(1.0, xNorm)),
        y: Math.max(0.0, Math.min(1.0, yNorm)),
        timestamp: timestampSec,
      } as any);
    };

    const webgazer = (window as any).webgazer as WebGazerLike | undefined;

    // Preferred path: use webgazer if available in the page.
    if (webgazer !== undefined) {
      webgazer
        .setGazeListener((data) => {
          const canvas = viewerMutable.canvas;
          if (!data || canvas === null) return;

          const rect = canvas.getBoundingClientRect();
          if (rect.width <= 0 || rect.height <= 0) return;

          const xNorm = (data.x - rect.left) / rect.width;
          const yNorm = (data.y - rect.top) / rect.height;
          sendNormalized(xNorm, yNorm, performance.now() / 1000.0);
        })
        .begin();

      return () => {
        if (webgazer.end) webgazer.end();
      };
    }

    // Fallback path: use pointer movement to validate the websocket pipeline.
    const onPointerMove = (event: PointerEvent) => {
      const canvas = viewerMutable.canvas;
      if (canvas === null) return;

      const rect = canvas.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return;

      const xNorm = (event.clientX - rect.left) / rect.width;
      const yNorm = (event.clientY - rect.top) / rect.height;
      sendNormalized(xNorm, yNorm, performance.now() / 1000.0);
    };

    window.addEventListener("pointermove", onPointerMove);
    return () => window.removeEventListener("pointermove", onPointerMove);
  }, [viewer.messageSource, viewerMutable]);

  return null;
}
