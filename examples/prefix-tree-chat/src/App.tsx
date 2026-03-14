import React, { useCallback, useEffect, useRef, useState } from 'react';
import { ModelManager, Wllama } from '@wllama/wllama';
import type { WllamaChatMessage } from '@wllama/wllama';
import wllamaSingle from '@wllama/wllama/src/single-thread/wllama.wasm?url';
import wllamaMulti from '@wllama/wllama/src/multi-thread/wllama.wasm?url';
import {
  DEFAULT_MEMORY_CAP_BYTES,
  PrefixTreeManagerImpl,
  buildTreeView,
} from './prefix-tree';
import type { ConversationNode, PrefixTreeState, TreeViewNode } from './types';

// ─────────────────────────────────────────────────────────────
// wllama worker 路径配置（与 examples/main 保持一致）
// ─────────────────────────────────────────────────────────────

const WLLAMA_CONFIG_PATHS = {
  'single-thread/wllama.wasm': wllamaSingle,
  'multi-thread/wllama.wasm': wllamaMulti,
};

// 固定本地模型文件名；实际 URL 在运行时基于当前页面地址生成
const LOCAL_MODEL_FILE = 'Llama-3.2-1B-Instruct-Q4_0.gguf';

// ─────────────────────────────────────────────────────────────
// 加载状态类型
// ─────────────────────────────────────────────────────────────

type LoadPhase = 'idle' | 'downloading' | 'loading' | 'ready' | 'error';

// ModelManager 单例（与 examples/main 一致，负责 OPFS 缓存管理）
const modelManager = new ModelManager();

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return '0 B';
  }

  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value >= 100 || unitIndex === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[unitIndex]}`;
}

function formatRelativeTime(timestamp: number): string {
  if (!timestamp) {
    return '未访问';
  }

  const deltaMs = Math.max(0, Date.now() - timestamp);
  if (deltaMs < 1000) return '刚刚';

  const seconds = Math.floor(deltaMs / 1000);
  if (seconds < 60) return `${seconds}s 前`;

  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m 前`;

  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h 前`;

  const days = Math.floor(hours / 24);
  return `${days}d 前`;
}

// ─────────────────────────────────────────────────────────────
// 顶层 App
// ─────────────────────────────────────────────────────────────

export default function App() {
  // ── 模型加载状态 ──────────────────────────────────────────
  const [loadPhase, setLoadPhase] = useState<LoadPhase>('idle');
  const [loadProgress, setLoadProgress] = useState(0); // 0~1
  const [loadError, setLoadError] = useState('');
  // 生成 worker 可解析的绝对 URL，兼容 Vite dev 与 base='./' 的静态部署
  const modelUrl = new URL(LOCAL_MODEL_FILE, window.location.href).href;
  const wllamaRef = useRef<Wllama | null>(null);
  const loadPromiseRef = useRef<Promise<void> | null>(null);
  const autoLoadStartedRef = useRef(false);

  // 固定本地模型加载逻辑
  const handleLoad = useCallback(async () => {
    if (loadPromiseRef.current) {
      await loadPromiseRef.current;
      return;
    }

    const loadTask = (async () => {
      if (loadPhase === 'ready') {
        return;
      }

      setLoadPhase('loading');
      setLoadProgress(0);
      setLoadError('');
      try {
        // 直接用本地 public/ 目录模型
        const model = await modelManager.downloadModel(modelUrl);
        const wllama = new Wllama(WLLAMA_CONFIG_PATHS, { preferWebGPU: false });
        await wllama.loadModel(model, { n_ctx: 4096 });
        wllamaRef.current = wllama;
        const manager = new PrefixTreeManagerImpl(wllama, {
          memoryCapBytes: DEFAULT_MEMORY_CAP_BYTES,
        });
        await manager.init(DEFAULT_MEMORY_CAP_BYTES);
        managerRef.current = manager;
        setHistory([]);
        setLoadPhase('ready');
        refresh();
      } catch (e) {
        setLoadError((e as Error).message ?? String(e));
        setLoadPhase('error');
      }
    })();

    loadPromiseRef.current = loadTask;
    try {
      await loadTask;
    } finally {
      loadPromiseRef.current = null;
    }
  }, [loadPhase, modelUrl]);

  useEffect(() => {
    if (loadPhase === 'idle' && !autoLoadStartedRef.current) {
      autoLoadStartedRef.current = true;
      void handleLoad();
    }
  }, [handleLoad, loadPhase]);

  // ── 对话树 ─────────────────────────────────────────────────
  const managerRef = useRef<PrefixTreeManagerImpl | null>(null);

  // 初始占位（模型未加载时用于渲染空 UI）
  const dummyMgrRef = useRef<PrefixTreeManagerImpl | null>(null);
  if (!dummyMgrRef.current) {
    // 传入一个带空实现的 dummy wllama，仅供 UI 渲染占位
    const dummyWllama = {} as Wllama;
    dummyMgrRef.current = new PrefixTreeManagerImpl(dummyWllama);
  }

  const mgr = managerRef.current ?? dummyMgrRef.current;

  const [treeState, setTreeState] = useState<PrefixTreeState>(mgr.getState());
  const [isGenerating, setIsGenerating] = useState(false);
  const [history, setHistory] = useState<WllamaChatMessage[]>([]);

  const refresh = useCallback(() => {
    const currentMgr = managerRef.current ?? dummyMgrRef.current;
    if (currentMgr) {
      setTreeState(currentMgr.getState());
    }
  }, []);

  // 应用层只维护 history，节点选择由 manager 在引擎状态中解析。
  let activeNodeId = treeState.rootId;
  try {
    activeNodeId = mgr.resolveNodeIdByHistory(history);
  } catch {
    activeNodeId = treeState.rootId;
  }
  // 当前激活节点的对象
  const activeNode = treeState.nodes.get(activeNodeId) ?? null;

  // 树视图
  const treeView = buildTreeView(treeState, activeNodeId);

  // ── 聊天输入 ───────────────────────────────────────────────

  const [input, setInput] = useState('');
  // 流式生成时的临时文本
  const [streamingText, setStreamingText] = useState('');
  const chatEndRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [history, streamingText]);

  const handleSend = useCallback(async () => {
    const msg = input.trim();
    if (!msg || isGenerating) return;
    const baseHistory = history;
    setInput('');
    setIsGenerating(true);
    setStreamingText('');

    abortRef.current = new AbortController();
    try {
      const nodeId = await mgr.chat(
        baseHistory,
        msg,
        (_piece: string, full: string) => {
          setStreamingText(full);
          refresh();
        },
        abortRef.current.signal
      );
      setHistory(mgr.getHistory(nodeId));
    } finally {
      setStreamingText('');
      setIsGenerating(false);
      refresh();
    }
  }, [history, input, isGenerating, mgr, refresh]);

  const handleStop = () => {
    abortRef.current?.abort();
  };

  const handlePickHistory = useCallback(
    (nodeId: number) => {
      if (isGenerating) return;
      setHistory(mgr.getHistory(nodeId));
      refresh();
    },
    [isGenerating, mgr, refresh]
  );

  const handleNewConversation = useCallback(() => {
    if (isGenerating) return;
    setHistory([]);
    setStreamingText('');
  }, [isGenerating]);

  useEffect(() => {
    return () => {
      const currentMgr = managerRef.current;
      if (currentMgr) {
        void currentMgr.finish();
      }
    };
  }, []);

  // ── 模型未就绪时渲染加载界面 ──────────────────────────────

  const isLoading = loadPhase === 'downloading' || loadPhase === 'loading';

  if (loadPhase !== 'ready') {
    return (
      <div style={styles.loadScreen}>
        <div style={styles.loadCard}>
          <h2 style={styles.loadTitle}>🌳 对话前缀树 KV Cache 演示</h2>
          <p style={styles.loadDesc}>
            加载 GGUF 模型后，即可在浏览器内运行 LLM 推理。<br />
            切换对话分支时，引擎自动复用公共前缀的 KV Cache，
            大幅减少重复计算。
          </p>

          <label style={styles.loadLabel}>固定测试模型</label>
          <div style={styles.loadInput}>{modelUrl}</div>

          {isLoading ? (
            <div style={styles.loadProgressWrap}>
              <div style={styles.loadProgressBar}>
                <div
                  style={{
                    ...styles.loadProgressFill,
                    width: loadPhase === 'loading' ? '100%' : `${Math.round(loadProgress * 100)}%`,
                    background: loadPhase === 'loading' ? '#7c3aed' : '#2563eb',
                  }}
                />
              </div>
              <span style={styles.loadProgressText}>
                {loadPhase === 'downloading'
                  ? `⬇ 下载中 ${Math.round(loadProgress * 100)}%…`
                  : '⚙ 加载模型进显存…'}
              </span>
            </div>
          ) : (
            <button
              style={styles.loadBtn}
              onClick={handleLoad}
              disabled={false}
            >
              重新加载模型
            </button>
          )}

          {loadPhase === 'error' && (
            <div style={styles.loadError}>
              ❌ 加载失败：{loadError}
            </div>
          )}
          <p style={styles.loadNote}>
            💡 当前页面会自动从本地静态文件加载模型，便于反复测试前缀树与 KV Cache 逻辑。
          </p>
        </div>
      </div>
    );
  }

  return (
    <div style={styles.root}>
      {/* ── 左侧：前缀树可视化 ── */}
      <aside style={styles.sidebar}>
        <div style={styles.sidebarHeader}>
          <span style={styles.sidebarTitle}>📂 对话前缀树</span>
          <button style={styles.resetBtn} onClick={handleNewConversation} disabled={isGenerating}>
            新会话
          </button>
        </div>
        <div style={styles.treeContainer}>
          {treeView.length === 0 ? (
            <div style={styles.emptyTree}>暂无对话，从右侧开始聊天</div>
          ) : (
            treeView.map((v) => (
              <TreeNode
                key={v.node.id}
                treeNode={v}
                onPickHistory={handlePickHistory}
                isGenerating={isGenerating}
              />
            ))
          )}
        </div>

        {/* KV Cache 状态面板 */}
        <KVCachePanel treeState={treeState} activeNodeId={activeNodeId} />
      </aside>

      {/* ── 右侧：主聊天区 ── */}
      <main style={styles.chatArea}>
        <div style={styles.chatHeader}>
          <span>💬 当前路径对话</span>
          <span style={styles.pathHint}>
            激活节点: #{activeNodeId}
            {activeNode && activeNode.prefixTokenCount >= 0 &&
              `  ·  前缀约 ${activeNode.prefixTokenCount} tokens`}
          </span>
        </div>

        <div style={styles.messageList}>
          {history.length === 0 && <div style={styles.emptyChat}>发送消息开始对话</div>}
          {history.map((msg, i) => (
            <MessageBubble key={i} role={msg.role} content={msg.content} />
          ))}
          {isGenerating && streamingText && (
            <MessageBubble role="assistant" content={streamingText} streaming />
          )}
          <div ref={chatEndRef} />
        </div>

        <div style={styles.inputArea}>
          <textarea
            style={styles.textarea}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="输入消息，Enter 发送，Shift+Enter 换行"
            rows={3}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            disabled={isGenerating}
          />
          <div style={styles.inputBtns}>
            {isGenerating ? (
              <button style={styles.stopBtn} onClick={handleStop}>
                ⏹ 停止
              </button>
            ) : (
              <button
                style={styles.sendBtn}
                onClick={handleSend}
                disabled={!input.trim()}
              >
                发送 ↵
              </button>
            )}
          </div>
        </div>

        {/* 同层历史快捷入口（仅切换 UI history，不直接控制引擎节点） */}
        <SiblingNav
          siblings={mgr.getSiblings(activeNodeId)}
          onPickHistory={handlePickHistory}
          isGenerating={isGenerating}
        />
      </main>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// 树节点组件
// ─────────────────────────────────────────────────────────────

function TreeNode({
  treeNode,
  onPickHistory,
  isGenerating,
}: {
  treeNode: TreeViewNode;
  onPickHistory: (id: number) => void;
  isGenerating: boolean;
}) {
  const { node, depth, isActive, isOnActivePath, children } = treeNode;
  const previewSource = node.turn.user || node.turn.assistant;
  const preview = previewSource.slice(0, 30) + (previewSource.length > 30 ? '…' : '');
  const accessText = formatRelativeTime(node.lastAccessedAt);

  const rowStyle: React.CSSProperties = {
    ...styles.treeRow,
    paddingLeft: 12 + depth * 16,
    background: isActive
      ? '#2563eb22'
      : isOnActivePath
      ? '#e0f2fe'
      : 'transparent',
    borderLeft: isActive ? '3px solid #2563eb' : isOnActivePath ? '3px solid #7dd3fc' : '3px solid transparent',
  };

  return (
    <div>
      <div style={rowStyle}>
        <span style={styles.treeRoleTag(false)}>
          QA
        </span>
        <div
          style={styles.treeTextCol}
          onClick={() => !isGenerating && onPickHistory(node.id)}
          title={`Q: ${node.turn.user}\n\nA: ${node.turn.assistant}`}
        >
          <span style={styles.treePreview}>
            {preview || <em style={{ color: '#aaa' }}>（生成中…）</em>}
          </span>
          <span style={styles.treeMeta}>
            {formatBytes(node.snapshotTokenBytes)} · {node.prefixTokenCount >= 0 ? `${node.prefixTokenCount}t` : '未缓存'} · {accessText}
          </span>
        </div>
        <span style={styles.treeNodeId}>#{node.id}</span>
        {node.status === 'generating' && <span style={styles.spinnerDot}>⏳</span>}
      </div>
      {children.length > 0 && (
        <div>
          {children.map((c) => (
            <TreeNode
              key={c.node.id}
              treeNode={c}
              onPickHistory={onPickHistory}
              isGenerating={isGenerating}
            />
          ))}
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// 消息气泡
// ─────────────────────────────────────────────────────────────

function MessageBubble({
  role,
  content,
  streaming,
}: {
  role: string;
  content: string;
  streaming?: boolean;
}) {
  const isUser = role === 'user';
  return (
    <div style={{ display: 'flex', justifyContent: isUser ? 'flex-end' : 'flex-start', marginBottom: 12 }}>
      <div
        style={{
          maxWidth: '75%',
          padding: '10px 14px',
          borderRadius: isUser ? '18px 18px 4px 18px' : '18px 18px 18px 4px',
          background: isUser ? '#2563eb' : '#f1f5f9',
          color: isUser ? '#fff' : '#1e293b',
          fontSize: 14,
          lineHeight: 1.6,
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
          boxShadow: '0 1px 3px #0001',
        }}
      >
        {content}
        {streaming && <span style={{ opacity: 0.5, animation: 'blink 1s step-end infinite' }}>▌</span>}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// 兄弟节点导航
// ─────────────────────────────────────────────────────────────

function SiblingNav({
  siblings,
  onPickHistory,
  isGenerating,
}: {
  siblings: ConversationNode[];
  onPickHistory: (id: number) => void;
  isGenerating: boolean;
}) {
  if (siblings.length === 0) return null;
  return (
    <div style={styles.siblingNav}>
      <span style={{ fontSize: 12, color: '#64748b', marginRight: 8 }}>
        🌿 平行分支：
      </span>
      {siblings.map((s) => (
        <button
          key={s.id}
          style={styles.siblingBtn}
          onClick={() => onPickHistory(s.id)}
          disabled={isGenerating}
          title={`Q: ${s.turn.user}\n\nA: ${s.turn.assistant}`}
        >
          #{s.id} 💬 {s.turn.user.slice(0, 12)}…
        </button>
      ))}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// KV Cache 状态面板
// ─────────────────────────────────────────────────────────────

function KVCachePanel({
  treeState,
  activeNodeId,
}: {
  treeState: PrefixTreeState;
  activeNodeId: number;
}) {
  const totalNodes = treeState.nodes.size - 1; // 减去根节点
  const cachedNodes = [...treeState.nodes.values()].filter(
    (n) => n.status === 'cached' && n.id !== 0
  ).length;
  const activeNode = treeState.nodes.get(activeNodeId);
  const activePrefixTokens = activeNode?.prefixTokenCount ?? 0;
  const activeSnapshotBytes = activeNode?.snapshotTokenBytes ?? 0;
  const usagePercent = treeState.memoryCapBytes > 0
    ? Math.min(100, (treeState.totalSnapshotTokenBytes / treeState.memoryCapBytes) * 100)
    : 0;

  return (
    <div style={styles.kvPanel}>
      <div style={styles.kvTitle}>🗂 KV Cache 状态</div>
      <div style={styles.kvRow}>
        <span>总节点数</span>
        <strong>{totalNodes}</strong>
      </div>
      <div style={styles.kvRow}>
        <span>已缓存节点</span>
        <strong style={{ color: '#16a34a' }}>{cachedNodes}</strong>
      </div>
      <div style={styles.kvRow}>
        <span>当前前缀 tokens</span>
        <strong style={{ color: '#2563eb' }}>{activePrefixTokens >= 0 ? activePrefixTokens : '—'}</strong>
      </div>
      <div style={styles.kvRow}>
        <span>当前节点快照字节</span>
        <strong style={{ color: '#0f766e' }}>{formatBytes(activeSnapshotBytes)}</strong>
      </div>
      <div style={styles.kvRow}>
        <span>快照总字节</span>
        <strong style={{ color: '#b45309' }}>{formatBytes(treeState.totalSnapshotTokenBytes)}</strong>
      </div>
      <div style={styles.kvRow}>
        <span>快照上界</span>
        <strong>{formatBytes(treeState.memoryCapBytes)}</strong>
      </div>
      <div style={styles.kvRow}>
        <span>Context/KV 实际分配</span>
        <strong>{formatBytes(treeState.contextMemoryBytes)}</strong>
      </div>
      <div style={styles.kvProgressBar}>
        <div style={{ ...styles.kvProgressFill, width: `${usagePercent}%` }} />
      </div>
      <div style={styles.kvProgressText}>{usagePercent.toFixed(1)}% of cap</div>
      <div style={styles.kvRow}>
        <span>最近 LRU 剪枝</span>
        <strong>
          {treeState.lastPrunedNodeIds.length > 0
            ? `#${treeState.lastPrunedNodeIds.join(', #')}`
            : '无'}
        </strong>
      </div>
      <div style={styles.kvNote}>
        ✅ 当前显示的是引擎真实字节数。KV/context 缓冲区本身是固定分配；可被 LRU 回收的是节点快照 token 数据，因此剪枝阈值也基于该值而不是前缀估算。
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// 样式
// ─────────────────────────────────────────────────────────────

const styles = {
  root: {
    display: 'flex',
    height: '100vh',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    background: '#f8fafc',
    color: '#1e293b',
    overflow: 'hidden',
  } as React.CSSProperties,

  sidebar: {
    width: 320,
    minWidth: 260,
    borderRight: '1px solid #e2e8f0',
    display: 'flex',
    flexDirection: 'column',
    background: '#fff',
    overflow: 'hidden',
  } as React.CSSProperties,

  sidebarHeader: {
    padding: '14px 16px',
    borderBottom: '1px solid #e2e8f0',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
  } as React.CSSProperties,

  sidebarTitle: {
    fontWeight: 600,
    fontSize: 15,
  } as React.CSSProperties,

  resetBtn: {
    fontSize: 12,
    padding: '4px 10px',
    border: '1px solid #e2e8f0',
    borderRadius: 6,
    background: '#fff',
    cursor: 'pointer',
    color: '#64748b',
  } as React.CSSProperties,

  treeContainer: {
    flex: 1,
    overflowY: 'auto',
    padding: '8px 0',
  } as React.CSSProperties,

  emptyTree: {
    padding: 20,
    textAlign: 'center',
    color: '#94a3b8',
    fontSize: 13,
  } as React.CSSProperties,

  treeRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    padding: '5px 12px 5px 12px',
    cursor: 'pointer',
    userSelect: 'none',
    transition: 'background .15s',
  } as React.CSSProperties,

  treeRoleTag: (isUser: boolean): React.CSSProperties => ({
    width: 18,
    height: 18,
    borderRadius: '50%',
    background: isUser ? '#2563eb' : '#16a34a',
    color: '#fff',
    fontSize: 10,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    flexShrink: 0,
  }),

  treePreview: {
    fontSize: 12,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    color: '#334155',
  } as React.CSSProperties,

  treeTextCol: {
    flex: 1,
    minWidth: 0,
    display: 'flex',
    flexDirection: 'column',
    gap: 2,
  } as React.CSSProperties,

  treeMeta: {
    fontSize: 10,
    color: '#64748b',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  } as React.CSSProperties,

  treeNodeId: {
    fontSize: 10,
    color: '#94a3b8',
    flexShrink: 0,
  } as React.CSSProperties,

  tokenCount: {
    fontSize: 10,
    color: '#7c3aed',
    flexShrink: 0,
    background: '#f3e8ff',
    borderRadius: 4,
    padding: '1px 4px',
  } as React.CSSProperties,

  spinnerDot: {
    fontSize: 12,
  } as React.CSSProperties,

  deleteBtn: {
    fontSize: 10,
    color: '#ef4444',
    border: 'none',
    background: 'transparent',
    cursor: 'pointer',
    padding: '2px 4px',
    borderRadius: 3,
    flexShrink: 0,
  } as React.CSSProperties,

  branchPanel: {
    borderTop: '1px solid #e2e8f0',
    padding: 12,
  } as React.CSSProperties,

  branchLabel: {
    fontSize: 12,
    color: '#475569',
    marginBottom: 8,
  } as React.CSSProperties,

  branchInputRow: {
    display: 'flex',
    gap: 6,
  } as React.CSSProperties,

  branchHint: {
    fontSize: 11,
    color: '#f59e0b',
    marginTop: 4,
  } as React.CSSProperties,

  kvPanel: {
    borderTop: '1px solid #e2e8f0',
    padding: 12,
    background: '#f8fafc',
  } as React.CSSProperties,

  kvTitle: {
    fontWeight: 600,
    fontSize: 12,
    marginBottom: 8,
    color: '#475569',
  } as React.CSSProperties,

  kvRow: {
    display: 'flex',
    justifyContent: 'space-between',
    fontSize: 12,
    marginBottom: 4,
    color: '#64748b',
  } as React.CSSProperties,

  kvNote: {
    marginTop: 8,
    fontSize: 11,
    color: '#f59e0b',
    lineHeight: 1.5,
  } as React.CSSProperties,

  kvProgressBar: {
    height: 8,
    borderRadius: 999,
    background: '#e2e8f0',
    overflow: 'hidden',
    margin: '8px 0 6px',
  } as React.CSSProperties,

  kvProgressFill: {
    height: '100%',
    borderRadius: 999,
    background: 'linear-gradient(90deg, #22c55e 0%, #eab308 60%, #ef4444 100%)',
  } as React.CSSProperties,

  kvProgressText: {
    fontSize: 11,
    color: '#64748b',
    marginBottom: 8,
  } as React.CSSProperties,

  chatArea: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
  } as React.CSSProperties,

  chatHeader: {
    padding: '14px 20px',
    borderBottom: '1px solid #e2e8f0',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    background: '#fff',
    fontWeight: 600,
    fontSize: 15,
  } as React.CSSProperties,

  pathHint: {
    fontSize: 12,
    color: '#7c3aed',
    fontWeight: 400,
  } as React.CSSProperties,

  messageList: {
    flex: 1,
    overflowY: 'auto',
    padding: '20px',
  } as React.CSSProperties,

  emptyChat: {
    textAlign: 'center',
    color: '#94a3b8',
    fontSize: 14,
    marginTop: 60,
  } as React.CSSProperties,

  inputArea: {
    padding: '12px 16px',
    borderTop: '1px solid #e2e8f0',
    background: '#fff',
    display: 'flex',
    gap: 10,
    alignItems: 'flex-end',
  } as React.CSSProperties,

  textarea: {
    flex: 1,
    resize: 'none',
    border: '1px solid #e2e8f0',
    borderRadius: 10,
    padding: '10px 14px',
    fontSize: 14,
    lineHeight: 1.5,
    outline: 'none',
    fontFamily: 'inherit',
  } as React.CSSProperties,

  input: {
    flex: 1,
    border: '1px solid #e2e8f0',
    borderRadius: 8,
    padding: '7px 10px',
    fontSize: 13,
    outline: 'none',
    fontFamily: 'inherit',
  } as React.CSSProperties,

  inputBtns: {
    display: 'flex',
    flexDirection: 'column',
    gap: 6,
  } as React.CSSProperties,

  sendBtn: {
    padding: '8px 16px',
    background: '#2563eb',
    color: '#fff',
    border: 'none',
    borderRadius: 8,
    cursor: 'pointer',
    fontSize: 13,
    fontWeight: 600,
    whiteSpace: 'nowrap',
  } as React.CSSProperties,

  stopBtn: {
    padding: '8px 16px',
    background: '#ef4444',
    color: '#fff',
    border: 'none',
    borderRadius: 8,
    cursor: 'pointer',
    fontSize: 13,
    fontWeight: 600,
    whiteSpace: 'nowrap',
  } as React.CSSProperties,

  siblingNav: {
    padding: '8px 16px',
    borderTop: '1px solid #e2e8f0',
    display: 'flex',
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: 6,
    background: '#fafafa',
  } as React.CSSProperties,

  siblingBtn: {
    fontSize: 11,
    padding: '3px 8px',
    border: '1px solid #e2e8f0',
    borderRadius: 6,
    background: '#fff',
    cursor: 'pointer',
    color: '#475569',
    maxWidth: 160,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  } as React.CSSProperties,

  // ── 加载界面 ────────────────────────────────────────────────
  loadScreen: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    height: '100vh',
    background: '#f8fafc',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
  } as React.CSSProperties,

  loadCard: {
    background: '#fff',
    borderRadius: 16,
    boxShadow: '0 4px 24px #0001',
    padding: '40px 48px',
    maxWidth: 540,
    width: '100%',
    display: 'flex',
    flexDirection: 'column',
    gap: 16,
  } as React.CSSProperties,

  loadTitle: {
    margin: 0,
    fontSize: 22,
    fontWeight: 700,
    color: '#1e293b',
  } as React.CSSProperties,

  loadDesc: {
    margin: 0,
    fontSize: 14,
    color: '#64748b',
    lineHeight: 1.7,
  } as React.CSSProperties,

  loadLabel: {
    fontSize: 13,
    fontWeight: 600,
    color: '#475569',
  } as React.CSSProperties,

  loadInput: {
    border: '1px solid #e2e8f0',
    borderRadius: 8,
    padding: '10px 14px',
    fontSize: 13,
    outline: 'none',
    fontFamily: 'inherit',
    color: '#1e293b',
  } as React.CSSProperties,

  loadProgressWrap: {
    display: 'flex',
    flexDirection: 'column',
    gap: 6,
  } as React.CSSProperties,

  loadProgressBar: {
    height: 8,
    borderRadius: 4,
    background: '#e2e8f0',
    overflow: 'hidden',
  } as React.CSSProperties,

  loadProgressFill: {
    height: '100%',
    background: '#2563eb',
    borderRadius: 4,
    transition: 'width .3s ease',
  } as React.CSSProperties,

  loadProgressText: {
    fontSize: 12,
    color: '#64748b',
  } as React.CSSProperties,

  loadBtn: {
    padding: '12px 0',
    background: '#2563eb',
    color: '#fff',
    border: 'none',
    borderRadius: 8,
    cursor: 'pointer',
    fontSize: 15,
    fontWeight: 600,
  } as React.CSSProperties,

  loadError: {
    fontSize: 13,
    color: '#ef4444',
    background: '#fef2f2',
    border: '1px solid #fecaca',
    borderRadius: 8,
    padding: '10px 14px',
  } as React.CSSProperties,

  loadNote: {
    margin: 0,
    fontSize: 12,
    color: '#94a3b8',
    lineHeight: 1.6,
  } as React.CSSProperties,
} as const;
