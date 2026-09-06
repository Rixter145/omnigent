import {
  memo,
  useCallback,
  useDeferredValue,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { useStickToBottomContext } from "use-stick-to-bottom";
import {
  Conversation,
  ConversationContent,
  ConversationEmptyState,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import { Message, MessageContent } from "@/components/ai-elements/message";
import { ElicitationCard } from "@/components/blocks/ApprovalCard";
import { cn } from "@/lib/utils";
import { getCurrentAuthorId } from "@/lib/identity";
import { hasCommandModifier } from "@/lib/hotkeys";
import {
  type Bubble,
  type BubbleCache,
  buildBubbles,
  createBubbleCache,
  liveCandidateAssistantIndex,
} from "@/lib/renderItems";
import { useChatStore } from "@/store/chatStore";
import { TranscriptScrollbar } from "@/pages/TranscriptScrollbar";
import { TurnRail, type Turn } from "@/pages/TurnRail";
import { StreamBudgetBanner } from "@/components/StreamBudgetBanner";
import { useUserMessageNav } from "@/hooks/useUserMessageNav";
import { ChatPlanAccordion } from "@/shell/ChatPlanAccordion";
import { RunnerStartingIndicator, McpStartupIndicator } from "@/pages/ChatIndicators";
import { CHAT_COLUMN_WIDTH } from "@/pages/chatLayout";
import {
  type ConversationScroller,
  BubbleView,
  ConversationScrollRefBridge,
  HistoryAutoLoader,
  HistoryLoadingIndicator,
  JumpToTopButton,
  KeepBottomOnViewportResize,
  LatestTurnSpacer,
  ScrollToBottomOnSend,
  UserMessageNavConnected,
  WorkingIndicator,
  bubbleKey,
  buildPendingBubbles,
  collectPendingElicitations,
  computeIsWorking,
  extractUserText,
  isSystemBubble,
  mergePendingBubbles,
  reorderCommittedRequestElicitations,
  shouldShowWorkingIndicator,
  stripGatedSubagentRoutingChips,
  stripPendingElicitations,
} from "@/components/chat/chatBubbleParts";

export interface TranscriptProps {
  /** Ref callback for the conversation wrapper element (SelectionPopup scope +
   *  JumpToTopButton hover ancestor). Owned by the parent, forwarded here. */
  setConversationEl: (el: HTMLDivElement | null) => void;
  /** Wrapper element the JumpToTopButton attaches its hover listeners to. */
  containerEl: HTMLElement | null;
  /** StickToBottom scroll container + lock controls, lifted by the bridge. */
  scroller: ConversationScroller | null;
  setScroller: (s: ConversationScroller | null) => void;
  /** Bumped on each local send so the list scrolls back to the bottom. */
  sendScrollNonce: number;
  hasMoreHistory: boolean;
  loadingMoreHistory: boolean;
  isMobileViewport: boolean;
  /** Display-only "Working…" gate (edge-driven, from the parent). */
  showsWorking: boolean;
  agentsError: unknown;
  /** True while a managed-sandbox launch is in flight (cold-launch spinner). */
  sandboxLaunching: boolean;
  /** Terminal-first spin-up bits for the cold-launch empty state. */
  terminalFirst: { isTerminalFirst: boolean; terminalStartingUp?: boolean } | null | undefined;
  /** Pub/sub ref for the LatestTurnSpacer's synchronous re-measure handle. */
  spacerMeasureRef: React.RefObject<(() => void) | null>;
}

/**
 * The scrolling transcript column: the ONLY subtree that subscribes to the
 * streaming-hot store fields (`blocks`, `activeResponse`, `pendingUserMessages`,
 * `interruptedResponseIds`, `sessionStatus`) and rebuilds the bubble list. It's
 * wrapped in `memo` and receives only edge-driven / stable props, so a streaming
 * frame re-renders this subtree alone — the composer, header, status bar, and
 * dialogs bail out via React's normal prop-equality check.
 */
function TranscriptImpl({
  setConversationEl,
  containerEl,
  scroller,
  setScroller,
  sendScrollNonce,
  hasMoreHistory,
  loadingMoreHistory,
  isMobileViewport,
  showsWorking,
  agentsError,
  sandboxLaunching,
  terminalFirst,
  spacerMeasureRef,
}: TranscriptProps) {
  const blocks = useChatStore((s) => s.blocks);
  const pendingUserMessages = useChatStore((s) => s.pendingUserMessages);
  const activeResponse = useChatStore((s) => s.activeResponse);
  const interruptedResponseIds = useChatStore((s) => s.interruptedResponseIds);
  const sessionStatus = useChatStore((s) => s.sessionStatus);
  const subagentRoutingOverride = useChatStore((s) => s.subagentRoutingOverride);
  const mcpStartupActive = useChatStore((s) => s.mcpStartup !== null);
  const hasTasks = useChatStore((s) => s.todos.length > 0);
  const conversationId = useChatStore((s) => s.conversationId);
  // Deferred with the SAME cadence as `listBubbles` below, so it changes in the
  // same committed render the new conversation's bubbles paint in. The scroll
  // reset keys on this (not the immediate id) so it restores position AFTER the
  // deferred content is in the DOM — pinning against the old content would land
  // mid-conversation once the deferred swap commits.
  const listConversationId = useDeferredValue(conversationId);

  // Build bubbles once per blocks/activeResponse change. Per-surface reuse
  // cache so a streaming append rebuilds only the active bubble, reusing the
  // finalized prefix by reference. Pending user messages (POSTed but not yet
  // acked) render as trailing user bubbles so the input is visible immediately.
  const bubbleCacheRef = useRef<BubbleCache>(createBubbleCache());
  const bubbles = useMemo<Bubble[]>(() => {
    const committed = stripGatedSubagentRoutingChips(
      reorderCommittedRequestElicitations(
        buildBubbles(
          blocks,
          activeResponse,
          bubbleCacheRef.current,
          interruptedResponseIds,
          computeIsWorking(sessionStatus),
        ),
      ),
      subagentRoutingOverride,
    );
    if (pendingUserMessages.length === 0) return committed;
    return mergePendingBubbles(
      committed,
      buildPendingBubbles(pendingUserMessages, getCurrentAuthorId()),
    );
  }, [
    blocks,
    activeResponse,
    interruptedResponseIds,
    pendingUserMessages,
    subagentRoutingOverride,
    sessionStatus,
  ]);

  // Virtualizer-derived geometry (scroll handle, active turn, range nonce),
  // published by VirtualBubbleList. The rail reads the active turn and the
  // spacer the range nonce, all from the virtualizer's model rather than the
  // windowed DOM. `scrollToItem` is held in a ref so `ensureItemVisible` keeps a
  // stable identity; the reactive fields are lifted to state.
  const scrollToItemRef = useRef<((itemId: string) => boolean) | null>(null);
  const [activeTurnId, setActiveTurnId] = useState<string | null>(null);
  const [spacerMeasureNonce, setSpacerMeasureNonce] = useState(0);
  const onGeometryChange = useCallback((geometry: TranscriptGeometry) => {
    scrollToItemRef.current = geometry.scrollToItem;
    setActiveTurnId(geometry.activeTurnId);
    setSpacerMeasureNonce(geometry.rangeNonce);
  }, []);
  const ensureItemVisible = useCallback(
    (itemId: string) => scrollToItemRef.current?.(itemId) ?? false,
    [],
  );

  // Single nav instance shared by hotkey + buttons. System-message bubbles are
  // excluded — the hotkey is for navigating real user turns, not markers.
  const userMessageIds = useMemo(
    () =>
      bubbles
        .filter(
          (b): b is Extract<Bubble, { kind: "user" }> => b.kind === "user" && !isSystemBubble(b),
        )
        .map((b) => b.itemId),
    [bubbles],
  );
  const nav = useUserMessageNav(userMessageIds, ensureItemVisible);

  // One rail tick per real user turn, paired with a preview of the reply that
  // followed. Mirrors the transcript's loaded window and grows lazily.
  const turns = useMemo<Turn[]>(() => {
    const out: Turn[] = [];
    for (let i = 0; i < bubbles.length; i++) {
      const b = bubbles[i];
      if (b.kind !== "user" || isSystemBubble(b)) continue;
      let preview = "";
      for (let j = i + 1; j < bubbles.length; j++) {
        const next = bubbles[j];
        if (next.kind === "user" && !isSystemBubble(next)) break;
        if (next.kind === "assistant") {
          const textItem = next.items.find((it) => it.kind === "text" && it.text.trim());
          if (textItem && textItem.kind === "text") {
            preview = textItem.text.trim();
            break;
          }
        }
      }
      out.push({
        itemId: b.itemId,
        userText: extractUserText(b.content),
        responsePreview: preview.slice(0, 240),
      });
    }
    return out;
  }, [bubbles]);

  // Pending elicitation cards float to the bottom of the chat: rendered as the
  // last items and removed from their inline slot so they don't render twice.
  // `streamBubbles` keeps `bubbles`' reference when nothing is pending.
  const pendingElicitations = useMemo(() => collectPendingElicitations(bubbles), [bubbles]);
  const streamBubbles = useMemo(
    () => (pendingElicitations.length === 0 ? bubbles : stripPendingElicitations(bubbles)),
    [bubbles, pendingElicitations.length],
  );
  // The windowed bubble list is the expensive render (markdown/tool subtrees).
  // Feed it a DEFERRED copy so a conversation switch — which swaps `blocks`
  // synchronously via the store — commits the cheap transcript shell (empty
  // state, indicators, rail, spacer) immediately and renders the heavy list at
  // transition priority, off the click frame. The Composer is a sibling outside
  // <Transcript>, so it is entirely unaffected. During steady-state streaming
  // React isn't busy, so useDeferredValue commits promptly (no token lag).
  // lastAssistantIndex is derived from the SAME deferred list so the live-turn
  // flag matches what the list actually renders.
  const listBubbles = useDeferredValue(streamBubbles);
  const lastAssistantIndex = useMemo(
    () => liveCandidateAssistantIndex(listBubbles),
    [listBubbles],
  );

  // Cmd+Alt+↑/↓ (Ctrl+Alt on win/linux) user-turn navigation.
  useEffect(() => {
    const handler = (e: globalThis.KeyboardEvent) => {
      if (!hasCommandModifier(e) || !e.altKey) return;
      if (e.key !== "ArrowUp" && e.key !== "ArrowDown") return;
      if (e.defaultPrevented) return;
      e.preventDefault();
      if (e.key === "ArrowUp") nav.goPrev();
      else nav.goNext();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [nav]);

  const showWorkingIndicator = shouldShowWorkingIndicator(showsWorking, bubbles);

  return (
    <>
      {/* Task tracker pinned above the thread. Sibling of the viewport (not an
      overlay) so it shrinks the scroll area rather than covering messages.
      Self-hides with no tasks. */}
      <ChatPlanAccordion className="mt-14 md:mt-12" />
      {/* Wrapper div gives us a ref to scope the SelectionPopup to the
      conversation area without requiring Conversation to forward refs. */}
      <div
        ref={setConversationEl}
        className="@container/chat relative flex min-h-0 flex-1 overflow-hidden"
      >
        <Conversation className={cn(!hasTasks && "chat-scroll-fade", "flex-1")}>
          <ConversationContent
            scrollClassName="transcript-hide-native-scrollbar"
            className={cn(
              "chat-conversation-content mx-auto w-full gap-4 px-4 pb-6",
              hasTasks ? "pt-4" : "pt-20",
              "md:pl-[clamp(1rem,(54rem-100cqi)*0.5+1rem,1.5rem)]",
              CHAT_COLUMN_WIDTH,
            )}
          >
            {/* Scroll helpers — must live inside StickToBottom to access context. */}
            {/* On a conversation switch the <Conversation> subtree is no longer
            remounted (that flashed scrollTop to 0 mid-teardown); instead this
            resets scroll position imperatively for the new conversation. Driven
            by the DEFERRED key, so the switch stays off the click frame. */}
            <ConversationSwitchReset conversationId={listConversationId} />
            <ScrollToBottomOnSend nonce={sendScrollNonce} />
            <KeepBottomOnViewportResize />
            <ConversationScrollRefBridge onScroller={setScroller} />
            <HistoryAutoLoader scrollElement={scroller?.el ?? null} />
            {bubbles.length === 0 && !showWorkingIndicator && !mcpStartupActive ? (
              (terminalFirst?.isTerminalFirst && terminalFirst.terminalStartingUp) ||
              sandboxLaunching ? (
                <RunnerStartingIndicator variant="hero" />
              ) : (
                <ConversationEmptyState>
                  <div className="space-y-1.5">
                    <h3 className="text-2xl font-medium tracking-[-0.02em]">
                      What should we work on?
                    </h3>
                    <p className="text-muted-foreground text-ui">
                      {agentsError
                        ? `Failed to load agents: ${agentsError instanceof Error ? agentsError.message : String(agentsError)}`
                        : "Send a message to get started."}
                    </p>
                  </div>
                </ConversationEmptyState>
              )
            ) : (
              <>
                {/* Older pages prepend here while their request is in flight. */}
                {loadingMoreHistory && <HistoryLoadingIndicator />}
                <VirtualBubbleList
                  bubbles={listBubbles}
                  scrollEl={scroller?.el ?? null}
                  lastAssistantIndex={lastAssistantIndex}
                  showsWorking={showsWorking}
                  listConversationId={listConversationId}
                  onGeometryChange={onGeometryChange}
                />
                {/* Pending elicitation cards, floated to the bottom of the chat
                so an outstanding question stays in view. Newest renders last,
                nearest the composer. Above the Working… indicator. */}
                {pendingElicitations.map((item) => (
                  <Message
                    key={item.elicitationId}
                    from="assistant"
                    className="max-w-full"
                    data-testid="bottom-elicitation"
                  >
                    <MessageContent className="w-full">
                      <ElicitationCard item={item} />
                    </MessageContent>
                  </Message>
                ))}
                {/* Working… shimmer, lit for the whole busy turn. */}
                {showWorkingIndicator && <WorkingIndicator />}
                {/* Terminal-first spin-up cue; self-gates to null off the
                spin-up window, and only when not already showing Working…. */}
                {!showWorkingIndicator && <RunnerStartingIndicator variant="row" />}
                {/* MCP-server startup band (codex-native); clears once the
                round settles (failures stay in host logs, not the chat). */}
                <McpStartupIndicator />
              </>
            )}
            {/* Frames the initially loaded turn at the top of the viewport. */}
            <LatestTurnSpacer
              scrollElement={scroller?.el ?? null}
              topGapPx={hasTasks ? 16 : undefined}
              measureRef={spacerMeasureRef}
              remeasureNonce={spacerMeasureNonce}
            />
          </ConversationContent>
          <ConversationScrollButton />
          <UserMessageNavConnected
            goPrev={nav.goPrev}
            goNext={nav.goNext}
            canPrev={nav.canPrev}
            canNext={nav.canNext}
            hidden={userMessageIds.length === 0}
          />
        </Conversation>
        {/* Constant-height scrollbar. Sibling of Conversation so it escapes the
        chat-scroll-fade mask. */}
        <TranscriptScrollbar scroller={scroller} topInset={hasTasks ? 12 : undefined} />
        {/* Hover the top edge to reveal a pill that loads all older history. */}
        <JumpToTopButton
          containerEl={containerEl}
          scroller={scroller}
          hasMoreHistory={hasMoreHistory}
        />
        {/* Too-many-tabs warning, a sibling of Conversation. */}
        <StreamBudgetBanner />
        {/* Left-edge minimap: one tick per turn. Desktop-only. */}
        {!isMobileViewport && (
          <TurnRail
            turns={turns}
            hasMoreHistory={hasMoreHistory}
            loadingMoreHistory={loadingMoreHistory}
            ensureItemVisible={ensureItemVisible}
            activeTurnId={activeTurnId}
          />
        )}
      </div>
    </>
  );
}

/**
 * The user turn that owns a given bubble index: the nearest user bubble at or
 * before it (a turn spans from its user bubble to the next). Falls back to the
 * first user turn when the index sits above every one. Pure so the active-tick
 * mapping is testable without the virtualizer, which supplies `midIndex` via
 * `getVirtualItemForOffset` — a lookup over ALL items, so it resolves a turn
 * whose row is windowed out above or below the viewport just the same.
 */
export function activeTurnIdAtBubbleIndex(bubbles: Bubble[], midIndex: number): string | null {
  const isUserTurn = (b: Bubble): b is Extract<Bubble, { kind: "user" }> =>
    b.kind === "user" && !isSystemBubble(b);
  if (bubbles.length === 0) return null;
  const start = Math.min(Math.max(midIndex, 0), bubbles.length - 1);
  for (let i = start; i >= 0; i--) {
    const b = bubbles[i];
    if (b && isUserTurn(b)) return b.itemId;
  }
  return bubbles.find(isUserTurn)?.itemId ?? null;
}

/**
 * Windows the bubble list off the StickToBottom scroll container so only the
 * on-screen slice mounts — switching into a long conversation no longer pays to
 * mount every bubble's markdown/tool subtree at once.
 *
 * The window is one flex child of the content div, sized to the full measured
 * height (`getTotalSize`) with each row absolutely positioned. Keeping the
 * wrapper at full height preserves `scrollHeight`, so StickToBottom's
 * at-bottom math, the TranscriptScrollbar, and the LatestTurnSpacer all keep
 * measuring the whole transcript even though most rows are unmounted. Rows are
 * keyed by `bubbleKey` so measured heights follow a bubble across a streaming
 * rebuild or a history prepend.
 */
/**
 * Transcript geometry derived from the virtualizer's model (all items, mounted
 * or not) — the single source of truth the rail and spacer read instead of
 * scanning the windowed DOM.
 */
export interface TranscriptGeometry {
  /** Pulls a turn's row into the mounted window; false if its id isn't found. */
  scrollToItem: (itemId: string) => boolean;
  /** itemId of the user turn owning the viewport midpoint, or null. */
  activeTurnId: string | null;
  /** Bumps whenever the mounted range changes, so the spacer can re-measure a
   *  windowed-out anchor that has since remounted. */
  rangeNonce: number;
}

/**
 * Where each conversation was last left, so returning restores it. `atBottom`
 * is stored as a flag rather than the pixel offset because "pinned to the
 * bottom" must survive the transcript re-measuring to a different height on
 * return — restoring a stale pixel would land mid-scroll. Written by
 * `ConversationSwitchReset` at the switch boundary (reading the live
 * scrollTop), read by it on the way back. Module-level so it outlives
 * re-renders; bounded so a long session doesn't retain every entry forever.
 */
interface TranscriptViewSnapshot {
  atBottom: boolean;
  offset: number;
}
const MAX_CACHED_VIEWS = 24;
const transcriptViewCache = new Map<string, TranscriptViewSnapshot>();
function rememberTranscriptView(convId: string, snap: TranscriptViewSnapshot): void {
  transcriptViewCache.delete(convId); // re-insert to refresh LRU order
  transcriptViewCache.set(convId, snap);
  while (transcriptViewCache.size > MAX_CACHED_VIEWS) {
    const oldest = transcriptViewCache.keys().next().value;
    if (oldest === undefined) break;
    transcriptViewCache.delete(oldest);
  }
}
/** Physical "is this scroll element at (or within a hair of) its bottom". */
const BOTTOM_EPSILON_PX = 8;
function isElAtBottom(el: HTMLElement): boolean {
  return el.scrollHeight - el.clientHeight - el.scrollTop <= BOTTOM_EPSILON_PX;
}
// The conversation whose scroll the reset is actively restoring. Its own
// programmatic scrollTop writes fire scroll events; without this the live save
// would record those transient values and clobber the real saved view (and flip
// its at-bottom flag), which the reset then reads back — a jitter feedback loop.
// The save skips while its conversation is being restored.
let restoringConversation: string | null = null;

/**
 * Restores the incoming conversation's scroll position on a switch, replacing
 * the full `<Conversation>` remount the deferred key used to force (which
 * flashed scrollTop to 0 as the subtree tore down). Lives inside
 * `<Conversation>` for the stick context. The saved view is written live by
 * VirtualBubbleList while the conversation is displayed; this only restores it.
 */
function ConversationSwitchReset({
  conversationId,
}: {
  conversationId: string | null | undefined;
}) {
  const ctx = useStickToBottomContext() as ReturnType<typeof useStickToBottomContext> & {
    scrollRef: React.RefObject<HTMLElement>;
    stopScroll: () => void;
  };
  // Read ctx through a ref so the effect fires ONCE per conversation change, not
  // whenever the stick context object gets a new identity (which re-ran the
  // restore repeatedly, each run reading a transient saved offset the prior
  // run's pin had written — jitter).
  const ctxRef = useRef(ctx);
  ctxRef.current = ctx;
  // `conversationId` is the DEFERRED id (see TranscriptImpl) — it changes in the
  // same committed render the new conversation's bubbles paint in, so this
  // restores against the freshly rendered content, not the outgoing one.
  useLayoutEffect(() => {
    const c = ctxRef.current;
    const el = c.scrollRef?.current;
    if (!el || !conversationId) return;
    // Suppress the live save for this conversation while we restore: our own
    // pin writes fire scroll events, and saving those transient values would
    // clobber the real saved view and feed a jitter loop.
    restoringConversation = conversationId;
    const saved = transcriptViewCache.get(conversationId);
    const done = () => {
      if (restoringConversation === conversationId) restoringConversation = null;
    };
    if (!saved || saved.atBottom) {
      // At bottom: hand it to StickToBottom's own lock, which is built to stay
      // pinned to the bottom through ALL later content growth (rows measuring,
      // the deferred list committing, late images) — no finite observer that
      // could disconnect a frame before one last growth and leave it short.
      // The deferred `conversationId` means we run after the new content is in
      // the DOM, so the lock has no stale frame to flash.
      c.scrollToBottom("instant");
      // Lift the save-suppression after the settle so genuine user scrolls
      // record again; the lock itself keeps pinning meanwhile.
      requestAnimationFrame(() => requestAnimationFrame(done));
      return;
    }
    // Scrolled up: release the lock and restore the saved pixel offset, holding
    // it across the settle (deferred commit + row measurement) with a short
    // read-free observer, then lift suppression.
    c.stopScroll();
    const content = el.firstElementChild as HTMLElement | null;
    const pin = () => {
      el.scrollTop = saved.offset;
    };
    pin();
    if (!content || typeof ResizeObserver === "undefined") {
      done();
      return;
    }
    let stable = 0;
    let last = -1;
    const observer = new ResizeObserver(([entry]) => {
      const h = entry?.contentRect.height ?? 0;
      pin();
      if (h === last) {
        if (++stable >= 3) {
          observer.disconnect();
          done();
        }
      } else {
        stable = 0;
        last = h;
      }
    });
    observer.observe(content);
    return () => {
      observer.disconnect();
      done();
    };
  }, [conversationId]);
  return null;
}

function VirtualBubbleList({
  bubbles,
  scrollEl,
  lastAssistantIndex,
  showsWorking,
  listConversationId,
  onGeometryChange,
}: {
  bubbles: Bubble[];
  scrollEl: HTMLElement | null;
  lastAssistantIndex: number;
  showsWorking: boolean;
  /** The conversation whose bubbles are CURRENTLY rendered (deferred id, in sync
   *  with `bubbles`). The scroll save keys on this so it attributes scroll to
   *  the displayed conversation and skips saves fired mid-switch (when the store
   *  has moved on but the deferred content — hence this id — hasn't caught up).*/
  listConversationId: string | null | undefined;
  /** Publishes virtualizer-derived geometry up to the rail/spacer. */
  onGeometryChange: (geometry: TranscriptGeometry) => void;
}) {
  // Save the displayed conversation's live scroll view (real scrollTop +
  // at-bottom flag) as the reader scrolls, so a later return restores it. Keyed
  // by `listConversationId` (the id of the content actually on screen) and
  // gated on the store having settled to it, so a scroll fired mid-switch — when
  // the store id leads the deferred content — can't be attributed to the wrong
  // conversation.
  const storeConvId = useChatStore((s) => s.conversationId);
  useEffect(() => {
    if (!scrollEl || !listConversationId) return;
    const save = () => {
      if (storeConvId !== listConversationId) return; // a switch is in flight
      if (restoringConversation === listConversationId) return; // our own pin, not a user scroll
      const snap = { atBottom: isElAtBottom(scrollEl), offset: Math.round(scrollEl.scrollTop) };
      rememberTranscriptView(listConversationId, snap);
    };
    save();
    scrollEl.addEventListener("scroll", save, { passive: true });
    return () => scrollEl.removeEventListener("scroll", save);
  }, [scrollEl, storeConvId, listConversationId]);

  const wrapperRef = useRef<HTMLDivElement>(null);
  // The list isn't the scroll container's first child — indicators, padding,
  // and the task tracker sit above it — so its top offset feeds the virtualizer
  // as scrollMargin. Without it every row's computed `start` is shifted and the
  // wrong window mounts. It goes stale whenever content ABOVE the list changes
  // height without changing `bubbles.length` (the history-loading indicator
  // toggling, the `pt-4 ↔ pt-20` task padding), so it is remeasured by watching
  // the content element — which reflows on any such change — not just the
  // scroll container (whose box size those changes leave untouched).
  const [scrollMargin, setScrollMargin] = useState(0);
  useLayoutEffect(() => {
    const wrapper = wrapperRef.current;
    if (!wrapper || !scrollEl) return;
    const measure = () => {
      const offset =
        wrapper.getBoundingClientRect().top -
        scrollEl.getBoundingClientRect().top +
        scrollEl.scrollTop;
      setScrollMargin((prev) => (Math.abs(prev - offset) >= 1 ? offset : prev));
    };
    measure();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(measure);
    observer.observe(scrollEl); // viewport height changes
    if (wrapper.parentElement) observer.observe(wrapper.parentElement); // content reflow above
    return () => observer.disconnect();
  }, [scrollEl, bubbles.length]);

  const virtualizer = useVirtualizer({
    count: bubbles.length,
    getScrollElement: () => scrollEl,
    // Corrected per row by measureElement; a middling bubble keeps the initial
    // total close enough that the first paint doesn't jump.
    estimateSize: () => 280,
    getItemKey: (index) => bubbleKey(bubbles[index]!),
    // Replaces the content column's `gap-4` between bubbles, which absolute
    // positioning would otherwise drop.
    gap: 16,
    overscan: 6,
    scrollMargin,
  });

  // Latest bubbles/virtualizer read through refs so published callbacks keep a
  // stable identity across renders.
  const bubblesRef = useRef(bubbles);
  bubblesRef.current = bubbles;
  const virtualizerRef = useRef(virtualizer);
  virtualizerRef.current = virtualizer;

  const scrollToItem = useCallback((itemId: string): boolean => {
    const index = bubblesRef.current.findIndex((b) => b.kind === "user" && b.itemId === itemId);
    if (index < 0) return false;
    virtualizerRef.current.scrollToIndex(index, { align: "center" });
    return true;
  }, []);


  const totalSize = virtualizer.getTotalSize();
  const range = virtualizer.range;

  // The user turn owning the viewport midpoint, from the virtualizer's model —
  // NOT the windowed DOM. `getVirtualItemForOffset` maps a scroll offset to a
  // bubble index across ALL items (mounted or not), so a turn whose row is
  // unmounted above OR below the viewport is still resolved correctly; the
  // active turn is the nearest user bubble at or before that index.
  const activeTurnId = useMemo(() => {
    const scrollOffset = virtualizer.scrollOffset;
    if (scrollOffset === null || bubbles.length === 0) return null;
    const viewport = scrollEl?.clientHeight ?? 0;
    const midItem = virtualizer.getVirtualItemForOffset(scrollOffset + viewport / 2);
    return activeTurnIdAtBubbleIndex(bubbles, midItem?.index ?? bubbles.length - 1);
    // `range` and `totalSize` are deps so the active turn recomputes as the
    // window scrolls and as measurements settle; `scrollOffset` alone isn't a
    // render trigger.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bubbles, scrollEl, range, totalSize, virtualizer]);

  // Older-history prepend compensation. Absolute rows are out of normal flow,
  // so the browser's native scroll anchoring — which `HistoryAutoLoader` leans
  // on to hold the read position across a prepend — can't act on them. Rather
  // than compensate by the total-height delta (which double-counts any change
  // ABOVE the list in the same commit, e.g. the HistoryLoadingIndicator being
  // removed), capture a still-present anchor row's offset before the prepend
  // and restore scrollTop so that row sits at the same viewport position after.
  // Measured purely from the virtualizer, so content above the list is
  // irrelevant. react-virtual then corrects the estimate→actual delta itself as
  // the freshly-mounted top rows measure.
  const prevFirstKeyRef = useRef<string | undefined>(undefined);
  // Snapshot of the previous render's first mounted row (top of the window,
  // nearest an incoming prepend): its key and the offset it held THEN. The
  // effect reads this (still the pre-prepend value) before overwriting it with
  // the current render's snapshot, so a prepend restores that row to the same
  // viewport position.
  const anchorSnapshotRef = useRef<{ key: string; offset: number } | null>(null);
  const firstVisible = virtualizer.getVirtualItems()[0];
  const currentSnapshot =
    firstVisible && typeof firstVisible.key === "string"
      ? { key: firstVisible.key, offset: firstVisible.start }
      : null;
  useLayoutEffect(() => {
    const firstKey = bubbles.length > 0 ? bubbleKey(bubbles[0]!) : undefined;
    const prevFirstKey = prevFirstKeyRef.current;
    const prevSnapshot = anchorSnapshotRef.current;
    prevFirstKeyRef.current = firstKey;
    anchorSnapshotRef.current = currentSnapshot;
    // Only a prepend (grew at the top, old first key still present). A switch
    // replaces the list (old first key gone) — handled by mount scroll-to-
    // bottom. Streaming appends leave the first key unchanged.
    if (!scrollEl || prevFirstKey === undefined || firstKey === prevFirstKey || !prevSnapshot) {
      return;
    }
    if (!bubbles.some((b) => bubbleKey(b) === prevFirstKey)) return;
    // Where the pre-prepend anchor row sits now, by key (its index shifted by
    // the prepend). Restore scrollTop so it holds its pre-prepend viewport
    // offset — measured purely from the virtualizer, so anything removed ABOVE
    // the list in the same commit (the HistoryLoadingIndicator) doesn't matter.
    const newIndex = bubbles.findIndex((b) => bubbleKey(b) === prevSnapshot.key);
    if (newIndex < 0) return;
    const newStart = virtualizer.getOffsetForIndex(newIndex, "start")?.[0];
    if (newStart === undefined) return;
    const delta = newStart - prevSnapshot.offset;
    if (delta > 0 && scrollEl.scrollTop > 1) scrollEl.scrollTop += delta;
    // currentSnapshot is intentionally captured per render; the effect only
    // needs the previous one, stored above.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bubbles, scrollEl, virtualizer]);

  // Publish geometry (scroll handle, active turn, range nonce) up to the rail
  // and spacer. rangeNonce bumps on any mounted-range change so a windowed-out
  // spacer anchor that has remounted gets a fresh measure.
  const rangeNonce = (range?.startIndex ?? -1) * 100003 + (range?.endIndex ?? -1);
  useEffect(() => {
    onGeometryChange({ scrollToItem, activeTurnId, rangeNonce });
  }, [onGeometryChange, scrollToItem, activeTurnId, rangeNonce]);

  return (
    <div ref={wrapperRef} className="relative w-full" style={{ height: `${totalSize}px` }}>
      {virtualizer.getVirtualItems().map((item) => {
        const bubble = bubbles[item.index];
        if (!bubble) return null;
        return (
          <div
            key={item.key}
            data-index={item.index}
            ref={virtualizer.measureElement}
            className="absolute top-0 left-0 w-full"
            style={{ transform: `translateY(${item.start - scrollMargin}px)` }}
          >
            <BubbleView
              bubble={bubble}
              isLastAssistant={item.index === lastAssistantIndex}
              showsWorking={showsWorking && item.index === lastAssistantIndex}
            />
          </div>
        );
      })}
    </div>
  );
}

export const Transcript = memo(TranscriptImpl);
