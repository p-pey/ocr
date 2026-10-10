/**
 * ConsensusReader (spec 8.5): multi-frame helper for live camera use.
 * Accept a date only when >= requiredVotes frames agree. Pure, no dependencies.
 * Defaults live in engineConfig.js §Q (CONSENSUS) — edit there, not here.
 */
import { CONSENSUS as CONSENSUS_CFG } from "./engineConfig.js";

export class ConsensusReader {
  constructor({ requiredVotes = CONSENSUS_CFG.requiredVotes, maxFrames = CONSENSUS_CFG.maxFrames } = {}) {
    this.requiredVotes = requiredVotes;
    this.maxFrames = maxFrames;
    this.votes = new Map(); // formatted -> { count, totalConfidence, lastBirthDate }
  }

  addFrameResult(result) {
    const formatted = result?.best?.birthDate?.formatted ?? null;
    if (!formatted) return this.status();
    const e =
      this.votes.get(formatted) ??
      { count: 0, totalConfidence: 0, birthDate: result.best.birthDate };
    e.count += 1;
    e.totalConfidence += result.best.birthDate.confidence ?? 0;
    e.birthDate = result.best.birthDate;
    this.votes.set(formatted, e);
    return this.status();
  }

  status() {
    let best = null;
    for (const [formatted, e] of this.votes) {
      if (!best || e.count > best.count) best = { formatted, ...e };
    }
    const agreed =
      best && best.count >= this.requiredVotes ? best : null;
    return { agreed, best, votes: Object.fromEntries(this.votes) };
  }

  reset() {
    this.votes.clear();
  }
}

/**
 * Convenience: run `recognize` on several image sources and return the first
 * date with >= requiredVotes agreements, else null.
 */
export async function recognizeWithConsensus(ocr, imageSources, { requiredVotes = CONSENSUS_CFG.requiredVotes } = {}) {
  const reader = new ConsensusReader({ requiredVotes });
  for (const src of imageSources) {
    const r = await ocr.recognize(src);
    const st = reader.addFrameResult(r);
    if (st.agreed) return { birthDate: st.agreed.birthDate, frames: reader.status() };
  }
  return { birthDate: null, frames: reader.status() };
}
