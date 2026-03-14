/**
 * ============================================================
 *  对话前缀树（Conversation Prefix Tree）核心类型定义
 * ============================================================
 *
 * 设计思路：
 *   每一次对话消息都作为树的一个节点，子节点代表在父节点消息之后
 *   继续的不同分支回复。所有从根节点到某个节点的路径，就构成了
 *   一段完整的对话历史（前缀）。
 *
 *   KV Cache 语义：
 *     - 每个节点持有 cachedTokens（已被 decode 进 KV Cache 的 token 序列）。
 *     - 当用户选择从节点 X "接续对话"时，引擎只需把 KV Cache 恢复到
 *       根→X 路径上的状态，然后继续 decode 新的输入，
 *       跳过对公共前缀重新计算的开销。
 *
 * ============================================================
 */

import type { WllamaChatMessage } from '@wllama/wllama';

export interface ConversationTurn {
  user: string;
  assistant: string;
}

// ─────────────────────────────────────────────────────────────
// 节点状态
// ─────────────────────────────────────────────────────────────

export type NodeStatus =
  | 'pending'    // 刚被创建，尚未推理
  | 'cached'     // KV Cache 已就绪（tokens 已 decode）
  | 'generating' // 正在生成回复
  | 'error';     // 推理出错

// ─────────────────────────────────────────────────────────────
// 对话节点
// ─────────────────────────────────────────────────────────────

export interface ConversationNode {
  /** 节点唯一 ID（自增） */
  id: number;

  /** 父节点 ID，根节点为 null */
  parentId: number | null;

  /** 子节点 ID 列表（每次分支对话都会产生新的子节点） */
  childIds: number[];

  /** 本节点携带的一轮对话（user + assistant） */
  turn: ConversationTurn;

  /** 本节点状态 */
  status: NodeStatus;

  /**
   * 从根节点到本节点（含）所对应的 token 序列长度。
   * 用于在恢复 KV Cache 时，快速定位需要保留的 prefix 长度。
   * -1 表示尚未计算。
   */
  prefixTokenCount: number;

  /** 生成耗时（ms），-1 表示未计算 */
  generationTimeMs: number;

  /** 本节点及其全部子树在 KV Cache 中的近似 token 占用 */
  cachedTokenCount: number;

  /** 当前节点快照 token 数据的真实字节数（host memory payload） */
  snapshotTokenBytes: number;

  /** 节点创建时间 */
  createdAt: number;

  /** 最近一次访问时间；访问某节点时，其所有祖先都会被更新 */
  lastAccessedAt: number;
}

// ─────────────────────────────────────────────────────────────
// 前缀树整体状态
// ─────────────────────────────────────────────────────────────

export interface PrefixTreeState {
  /** 所有节点的 Map（id → node） */
  nodes: Map<number, ConversationNode>;

  /**
   * 虚拟根节点 ID（id=0），不携带实际消息，
   * 代表"空对话"起始点，所有第一轮 user 消息都是它的子节点。
   */
  rootId: number;

  /** 当前"激活"节点 ID（用于接续对话的基准） */
  activeNodeId: number;

  /** 下一个分配的节点 ID */
  nextId: number;

  /** llama.cpp 上下文/KV 缓冲区的真实分配字节数 */
  contextMemoryBytes: number;

  /** 前缀树允许占用的估算显存上界（bytes） */
  memoryCapBytes: number;

  /** 当前所有节点快照 token 数据的真实字节总和（bytes） */
  totalSnapshotTokenBytes: number;

  /** 最近一次 LRU 剪枝删除的节点 ID 列表 */
  lastPrunedNodeIds: number[];

  /** 最近一次 LRU 剪枝发生时间；0 表示尚未发生 */
  lastPrunedAt: number;
}

// ─────────────────────────────────────────────────────────────
// 接口：ConversationPrefixTreeManager
//
// 这是应用层与推理引擎之间的核心接口。
// 应用层只需调用这几个方法，底层负责维护 KV Cache 的一致性。
// ─────────────────────────────────────────────────────────────

export interface ConversationPrefixTreeManager {
  /**
   * 获取当前前缀树状态（只读快照）。
   */
  getState(): PrefixTreeState;

  /**
   * 获取从根节点到指定节点的完整消息路径（对话历史）。
   * @param nodeId 目标节点 ID
   * @returns 有序的 WllamaChatMessage 数组
   */
  getHistory(nodeId: number): WllamaChatMessage[];

  /**
   * 基于应用层传入的 history 继续对话。
   *
   * 约定：
   *   - 应用层只传 messages，不传树节点 ID；
   *   - 引擎层根据 history 自动定位公共前缀并决定分支位置。
   *
   * @param history 作为上下文的完整历史消息
   * @param userMessage 本轮用户输入
   * @param onToken 流式回调，每生成一个 token 触发一次
   * @param abortSignal 可选的中止信号
   * @returns 新创建的 assistant 节点 ID
   */
  chat(
    history: WllamaChatMessage[],
    userMessage: string,
    onToken: (piece: string, fullText: string) => void,
    abortSignal?: AbortSignal
  ): Promise<number>;

  /**
   * 根据 history 解析出当前对应的节点 ID。
   * 用于 UI 高亮/导航，不要求应用层直接操纵树结构。
   */
  resolveNodeIdByHistory(history: WllamaChatMessage[]): number;

  /**
   * 获取指定节点的所有"兄弟节点"（同一 parent 的其他子节点）。
   * 用于 UI 展示当前节点有哪些平行分支。
   */
  getSiblings(nodeId: number): ConversationNode[];

  /**
   * 结束一次会话（finish）。
   * 默认行为是重置当前会话树，让后续聊天从空 history 开始。
   */
  finish(): Promise<void>;
}

// ─────────────────────────────────────────────────────────────
// UI 层辅助类型
// ─────────────────────────────────────────────────────────────

/** 用于渲染树视图的节点展开结构 */
export interface TreeViewNode {
  node: ConversationNode;
  depth: number;
  isActive: boolean;
  isOnActivePath: boolean;
  children: TreeViewNode[];
}

/** 应用 UI 状态 */
export interface AppState {
  /** 模型加载进度 0~1，-1=未开始 */
  loadProgress: number;
  /** 模型是否已就绪 */
  modelReady: boolean;
  /** 是否正在生成 */
  isGenerating: boolean;
  /** 错误信息 */
  error: string | null;
}
