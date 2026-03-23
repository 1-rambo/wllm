/**
 * 对话树管理器：应用层只做状态展示与调用编排。
 * 树维护、KV 快照、LRU 剪枝和内存上限控制全部在引擎层完成。
 */

import type {
  Wllama,
  WllamaChatMessage,
  WllamaTieredCacheOptions,
  WllamaTreeState,
} from '@wllama/wllama';
import type {
  ConversationNode,
  ConversationPrefixTreeManager,
  ConversationTurn,
  NodeStatus,
  PrefixTreeState,
  TreeViewNode,
} from './types';

export interface PrefixTreeManagerOptions {
  memoryCapBytes?: number;
  tieredCache?: WllamaTieredCacheOptions;
}

export const DEFAULT_MEMORY_CAP_BYTES = 1024 * 1024 * 1024;

// ─────────────────────────────────────────────────────────────
// 工厂函数
// ─────────────────────────────────────────────────────────────

function makeRootNode(): ConversationNode {
  const now = Date.now();
  return {
    id: 0,
    parentId: null,
    childIds: [],
    turn: { user: '', assistant: '' },
    status: 'cached',
    prefixTokenCount: 0,
    generationTimeMs: 0,
    cachedTokenCount: 0,
    snapshotTokenBytes: 0,
    createdAt: now,
    lastAccessedAt: now,
  };
}

// ─────────────────────────────────────────────────────────────
// PrefixTreeManagerImpl
// ─────────────────────────────────────────────────────────────

export class PrefixTreeManagerImpl implements ConversationPrefixTreeManager {
  private wllama: Wllama;
  private state: PrefixTreeState;
  private initialized = false;
  private options: PrefixTreeManagerOptions;

  constructor(wllama: Wllama, options: PrefixTreeManagerOptions = {}) {
    this.wllama = wllama;
    this.options = options;
    const root = makeRootNode();
    this.state = {
      nodes: new Map([[0, root]]),
      rootId: 0,
      activeNodeId: 0,
      nextId: 1,
      contextMemoryBytes: 0,
      memoryCapBytes: options.memoryCapBytes ?? DEFAULT_MEMORY_CAP_BYTES,
      totalSnapshotTokenBytes: 0,
      lastPrunedNodeIds: [],
      lastPrunedAt: 0,
      tieredCacheEnabled: options.tieredCache?.enabled ?? false,
      tierStats: {
        l1Tokens: 0,
        l2Tokens: 0,
        l3Tokens: 0,
        l1Slots: 0,
        l2Slots: 0,
        l3Slots: 0,
        promotions: 0,
        demotions: 0,
        diskReads: 0,
        diskWrites: 0,
        l3OverflowEvents: 0,
      },
    };
  }

  async init(
    memoryCapBytes: number = this.state.memoryCapBytes,
    tieredCache: WllamaTieredCacheOptions = this.options.tieredCache ?? {}
  ): Promise<void> {
    this.options = {
      ...this.options,
      memoryCapBytes,
      tieredCache,
    };
    const state = await this.wllama.chatSessionInit(memoryCapBytes, tieredCache);
    this.applyEngineState(state);
    this.initialized = true;
  }

  // ── 只读访问 ────────────────────────────────────────────────

  getState(): PrefixTreeState {
    return {
      ...this.state,
      nodes: new Map(
        [...this.state.nodes.entries()].map(([id, node]) => [id, {
          ...node,
          childIds: [...node.childIds],
          turn: { ...node.turn },
        }])
      ),
      lastPrunedNodeIds: [...this.state.lastPrunedNodeIds],
    };
  }

  getHistory(nodeId: number): WllamaChatMessage[] {
    return this.getPathToNode(nodeId)
      .filter((n) => n.id !== this.state.rootId)
      .flatMap((n) => {
        const history: WllamaChatMessage[] = [];
        if (n.turn.user) {
          history.push({ role: 'user', content: n.turn.user });
        }
        if (n.turn.assistant) {
          history.push({ role: 'assistant', content: n.turn.assistant });
        }
        return history;
      });
  }

  getSiblings(nodeId: number): ConversationNode[] {
    const node = this.requireNode(nodeId);
    if (node.parentId === null) return [];
    const parent = this.requireNode(node.parentId);
    return parent.childIds
      .filter((cid) => cid !== nodeId)
      .map((cid) => this.requireNode(cid));
  }

  // ── 对话生成 ────────────────────────────────────────────────

  async chat(
    history: WllamaChatMessage[],
    userMessage: string,
    onToken: (piece: string, fullText: string) => void,
    abortSignal?: AbortSignal
  ): Promise<number> {
    await this.ensureInitialized();
    const result = await this.wllama.chatSessionChat(history, userMessage, {
      stream: true,
      useCache: true,
      nPredict: Number.POSITIVE_INFINITY,
      sampling: { temp: 0.7, top_p: 0.9 },
      abortSignal,
      onChunk: onToken,
    });

    this.applyEngineState(result.state);
    return result.nodeId;
  }

  async chatFromNodeId(
    parentNodeId: number,
    userMessage: string,
    onToken: (piece: string, fullText: string) => void,
    abortSignal?: AbortSignal
  ): Promise<number> {
    await this.ensureInitialized();
    const result = await this.wllama.chatFromNode(parentNodeId, userMessage, {
      stream: true,
      useCache: true,
      nPredict: Number.POSITIVE_INFINITY,
      sampling: { temp: 0.7, top_p: 0.9 },
      abortSignal,
      onChunk: onToken,
    });

    this.applyEngineState(result.state);
    return result.nodeId;
  }

  async finish(): Promise<void> {
    await this.ensureInitialized();
    const state = await this.wllama.chatSessionFinish();
    this.applyEngineState(state);
  }

  resolveNodeIdByHistory(history: WllamaChatMessage[]): number {
    const turns = this.toTurns(history);
    let currentId = this.state.rootId;

    for (const turn of turns) {
      const current = this.requireNode(currentId);
      const nextId = current.childIds.find((childId) => {
        const child = this.state.nodes.get(childId);
        return !!child
          && child.turn.user === turn.user
          && child.turn.assistant === turn.assistant;
      });

      if (nextId === undefined) {
        throw new Error('History does not map to an existing cached conversation path');
      }
      currentId = nextId;
    }

    return currentId;
  }

  // ─────────────────────────────────────────────────────────────
  // 私有辅助
  // ─────────────────────────────────────────────────────────────

  private requireNode(id: number): ConversationNode {
    const node = this.state.nodes.get(id);
    if (!node) throw new Error(`Node ${id} not found`);
    return node;
  }

  private getPathToNode(nodeId: number): ConversationNode[] {
    const path: ConversationNode[] = [];
    let cur: ConversationNode | undefined = this.requireNode(nodeId);
    while (cur) {
      path.unshift(cur);
      if (cur.parentId === null) break;
      cur = this.state.nodes.get(cur.parentId);
    }
    return path;
  }

  private async ensureInitialized(): Promise<void> {
    if (this.initialized) {
      return;
    }
    await this.init(this.state.memoryCapBytes);
  }

  private toTurns(history: WllamaChatMessage[]): ConversationTurn[] {
    const turns: ConversationTurn[] = [];
    let pendingUser = '';

    for (const msg of history) {
      if (msg.role === 'system') {
        continue;
      }
      if (msg.role === 'user') {
        if (pendingUser) {
          throw new Error('Invalid history: consecutive user messages are not supported');
        }
        pendingUser = msg.content;
        continue;
      }
      if (msg.role === 'assistant') {
        if (!pendingUser) {
          throw new Error('Invalid history: assistant message without preceding user message');
        }
        turns.push({ user: pendingUser, assistant: msg.content });
        pendingUser = '';
      }
    }

    if (pendingUser) {
      throw new Error('Invalid history: trailing user message is not allowed in base history');
    }

    return turns;
  }

  private applyEngineState(state: WllamaTreeState): void {
    const nodes = new Map<number, ConversationNode>();
    for (const [id, node] of state.nodes.entries()) {
      nodes.set(id, {
        id: node.id,
        parentId: node.parentId,
        childIds: [...node.childIds],
        turn: {
          user: node.turn.user,
          assistant: node.turn.assistant,
        },
        status: (['pending', 'cached', 'generating', 'error'].includes(node.status)
          ? node.status
          : 'pending') as NodeStatus,
        prefixTokenCount: node.prefixTokenCount,
        generationTimeMs: node.generationTimeMs,
        cachedTokenCount: node.cachedTokenCount,
        snapshotTokenBytes: node.snapshotTokenBytes,
        createdAt: node.createdAt,
        lastAccessedAt: node.lastAccessedAt,
      });
    }

    this.state = {
      nodes,
      rootId: state.rootId,
      activeNodeId: state.activeNodeId,
      nextId: state.nextId,
      contextMemoryBytes: state.contextMemoryBytes,
      memoryCapBytes: state.memoryCapBytes,
      totalSnapshotTokenBytes: state.totalSnapshotTokenBytes,
      lastPrunedNodeIds: [...state.lastPrunedNodeIds],
      lastPrunedAt: state.lastPrunedAt,
      tieredCacheEnabled: state.tieredCacheEnabled,
      tierStats: {
        ...state.tierStats,
      },
    };
  }
}

// ─────────────────────────────────────────────────────────────
// UI 辅助
// ─────────────────────────────────────────────────────────────

export function buildTreeView(
  state: PrefixTreeState,
  activeNodeId: number
): TreeViewNode[] {
  const activePath = getActivePath(state, activeNodeId);
  const activePathSet = new Set(activePath.map((n) => n.id));

  function buildNode(nodeId: number, depth: number): TreeViewNode {
    const node = state.nodes.get(nodeId)!;
    return {
      node,
      depth,
      isActive: node.id === activeNodeId,
      isOnActivePath: activePathSet.has(node.id),
      children: node.childIds.map((cid) => buildNode(cid, depth + 1)),
    };
  }

  const root = state.nodes.get(state.rootId)!;
  return root.childIds.map((cid) => buildNode(cid, 0));
}

function getActivePath(
  state: PrefixTreeState,
  activeNodeId: number
): ConversationNode[] {
  const path: ConversationNode[] = [];
  let cur = state.nodes.get(activeNodeId);
  while (cur) {
    path.unshift(cur);
    if (cur.parentId === null) break;
    cur = state.nodes.get(cur.parentId);
  }
  return path;
}
