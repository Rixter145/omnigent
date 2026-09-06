// Client-only conversation ids used by the navigate-first "instant new chat"
// flow: while `createSession` is in flight the UI mints a `temp:<hex>` id,
// paints the optimistic first message and navigates into it, then rekeys to the
// real server id once the create returns.
//
// A temp id is a routing/registry token only — it never names a server session,
// so every server-scoped session hook must treat it as "no session yet". This
// lives in a leaf module (no store/React imports) so those low-level query hooks
// can enforce that by construction, without an import cycle back to the store.

/** Prefix of a client-only conversation id. */
export const TEMP_CONV_ID_PREFIX = "temp:";

/** Whether an id is a client-only temp conversation id (not a server session). */
export function isTempConvId(id: string | null | undefined): boolean {
  return typeof id === "string" && id.startsWith(TEMP_CONV_ID_PREFIX);
}

/** Mint a fresh client-only conversation id: `temp:<32-bit hex>`. */
export function newTempConvId(): string {
  const hex = Math.floor(Math.random() * 0x1_0000_0000)
    .toString(16)
    .padStart(8, "0");
  return `${TEMP_CONV_ID_PREFIX}${hex}`;
}
