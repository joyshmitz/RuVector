import { create } from 'zustand';
import type { NetworkStats, NodeInfo, TimeCrystal, CreditBalance } from '../types';
import { edgeNetService } from '../services/edgeNet';
import { storageService } from '../services/storage';

interface ContributionSettings {
  enabled: boolean;
  cpuLimit: number;
  gpuEnabled: boolean;
  gpuLimit: number;
  memoryLimit: number;
  bandwidthLimit: number;
  respectBattery: boolean;
  onlyWhenIdle: boolean;
  idleThreshold: number;
  consentGiven: boolean;
  consentTimestamp: Date | null;
}

interface NetworkState {
  stats: NetworkStats;
  nodes: NodeInfo[];
  timeCrystal: TimeCrystal;
  credits: CreditBalance;
  isConnected: boolean;
  isLoading: boolean;
  error: string | null;
  startTime: number;
  contributionSettings: ContributionSettings;
  isWASMReady: boolean;
  nodeId: string | null;
  // Persisted cumulative values from IndexedDB
  persistedCredits: number;
  persistedTasks: number;
  persistedUptime: number;

  setStats: (stats: Partial<NetworkStats>) => void;
  addNode: (node: NodeInfo) => void;
  removeNode: (nodeId: string) => void;
  updateNode: (nodeId: string, updates: Partial<NodeInfo>) => void;
  setTimeCrystal: (crystal: Partial<TimeCrystal>) => void;
  setCredits: (credits: Partial<CreditBalance>) => void;
  setConnected: (connected: boolean) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  updateRealStats: () => void;
  getUptime: () => number;
  setContributionSettings: (settings: Partial<ContributionSettings>) => void;
  giveConsent: () => void;
  revokeConsent: () => void;
  initializeEdgeNet: () => Promise<void>;
  startContributing: () => void;
  stopContributing: () => void;
  saveToIndexedDB: () => Promise<void>;
  loadFromIndexedDB: () => Promise<void>;
}

const initialStats: NetworkStats = {
  totalNodes: 0,
  activeNodes: 0,
  totalCompute: 0,
  creditsEarned: 0,
  tasksCompleted: 0,
  uptime: 0,
  latency: 0,
  bandwidth: 0,
};

const initialTimeCrystal: TimeCrystal = {
  phase: 0,
  frequency: 1.618,
  coherence: 0,
  entropy: 1.0,
  synchronizedNodes: 0,
};

const initialCredits: CreditBalance = {
  available: 0,
  pending: 0,
  earned: 0,
  spent: 0,
};

const defaultContributionSettings: ContributionSettings = {
  enabled: false,
  cpuLimit: 50,
  gpuEnabled: false,
  gpuLimit: 30,
  memoryLimit: 512,
  bandwidthLimit: 10,
  respectBattery: true,
  onlyWhenIdle: true,
  idleThreshold: 30,
  consentGiven: false,
  consentTimestamp: null,
};

export const useNetworkStore = create<NetworkState>()((set, get) => ({
  stats: initialStats,
  nodes: [],
  timeCrystal: initialTimeCrystal,
  credits: initialCredits,
  isConnected: false,
  isLoading: true,
  error: null,
  startTime: Date.now(),
  contributionSettings: defaultContributionSettings,
  isWASMReady: false,
  nodeId: null,
  persistedCredits: 0,
  persistedTasks: 0,
  persistedUptime: 0,

  setStats: (stats) =>
    set((state) => ({ stats: { ...state.stats, ...stats } })),

  addNode: (node) =>
    set((state) => {
      const newNodes = [...state.nodes, node];
      return {
        nodes: newNodes,
        stats: {
          ...state.stats,
          totalNodes: newNodes.length,
          activeNodes: newNodes.filter((n) => n.status === 'active').length,
        },
      };
    }),

  removeNode: (nodeId) =>
    set((state) => {
      const newNodes = state.nodes.filter((n) => n.id !== nodeId);
      return {
        nodes: newNodes,
        stats: {
          ...state.stats,
          totalNodes: newNodes.length,
          activeNodes: newNodes.filter((n) => n.status === 'active').length,
        },
      };
    }),

  updateNode: (nodeId, updates) =>
    set((state) => ({
      nodes: state.nodes.map((n) =>
        n.id === nodeId ? { ...n, ...updates } : n
      ),
    })),

  setTimeCrystal: (crystal) =>
    set((state) => ({
      timeCrystal: { ...state.timeCrystal, ...crystal },
    })),

  setCredits: (credits) =>
    set((state) => ({
      credits: { ...state.credits, ...credits },
    })),

  setConnected: (connected) =>
    set({ isConnected: connected, isLoading: false }),

  setLoading: (loading) => set({ isLoading: loading }),

  setError: (error) => set({ error, isLoading: false }),

  getUptime: () => {
    const state = get();
    return (Date.now() - state.startTime) / 1000;
  },

  setContributionSettings: (settings) =>
    set((state) => ({
      contributionSettings: { ...state.contributionSettings, ...settings },
    })),

  loadFromIndexedDB: async () => {
    try {
      const savedState = await storageService.loadState();
      if (savedState) {
        set({
          persistedCredits: savedState.creditsEarned,
          persistedTasks: savedState.tasksCompleted,
          persistedUptime: savedState.totalUptime,
          nodeId: savedState.nodeId,
          contributionSettings: {
            ...defaultContributionSettings,
            consentGiven: savedState.consentGiven,
            consentTimestamp: savedState.consentTimestamp
              ? new Date(savedState.consentTimestamp)
              : null,
            cpuLimit: savedState.cpuLimit,
            gpuEnabled: savedState.gpuEnabled,
            gpuLimit: savedState.gpuLimit,
            respectBattery: savedState.respectBattery,
            onlyWhenIdle: savedState.onlyWhenIdle,
          },
          credits: {
            earned: savedState.creditsEarned,
            spent: savedState.creditsSpent,
            available: savedState.creditsEarned - savedState.creditsSpent,
            pending: 0,
          },
          stats: {
            ...initialStats,
            creditsEarned: savedState.creditsEarned,
            tasksCompleted: savedState.tasksCompleted,
          },
        });
        console.log('[EdgeNet] Loaded persisted state:', savedState.creditsEarned, 'rUv');
      }
    } catch (error) {
      console.error('[EdgeNet] Failed to load from IndexedDB:', error);
    }
  },

  saveToIndexedDB: async () => {
    const state = get();
    try {
      await storageService.saveState({
        id: 'primary',
        nodeId: state.nodeId,
        creditsEarned: state.credits.earned,
        creditsSpent: state.credits.spent,
        tasksCompleted: state.stats.tasksCompleted,
        tasksSubmitted: 0,
        totalUptime: state.stats.uptime + state.persistedUptime,
        lastActiveTimestamp: Date.now(),
        consentGiven: state.contributionSettings.consentGiven,
        consentTimestamp: state.contributionSettings.consentTimestamp?.getTime() || null,
        cpuLimit: state.contributionSettings.cpuLimit,
        gpuEnabled: state.contributionSettings.gpuEnabled,
        gpuLimit: state.contributionSettings.gpuLimit,
        respectBattery: state.contributionSettings.respectBattery,
        onlyWhenIdle: state.contributionSettings.onlyWhenIdle,
      });
    } catch (error) {
      console.error('[EdgeNet] Failed to save to IndexedDB:', error);
    }
  },

  giveConsent: () => {
    set((state) => ({
      contributionSettings: {
        ...state.contributionSettings,
        consentGiven: true,
        consentTimestamp: new Date(),
      },
    }));
    get().saveToIndexedDB();
    console.log('[EdgeNet] User consent given for contribution');
  },

  revokeConsent: async () => {
    const { stopContributing } = get();
    stopContributing();
    set((state) => ({
      contributionSettings: {
        ...state.contributionSettings,
        consentGiven: false,
        consentTimestamp: null,
        enabled: false,
      },
    }));
    await storageService.clear();
    console.log('[EdgeNet] User consent revoked, data cleared');
  },

  initializeEdgeNet: async () => {
    try {
      set({ isLoading: true, error: null });
      console.log('[EdgeNet] Initializing...');

      // Load persisted state from IndexedDB first
      await get().loadFromIndexedDB();

      // Initialize WASM module
      await edgeNetService.init();
      const isWASMReady = edgeNetService.isWASMAvailable();
      set({ isWASMReady });

      if (isWASMReady) {
        console.log('[EdgeNet] WASM module ready');
        const node = await edgeNetService.createNode();
        if (node) {
          const nodeId = node.nodeId();
          set({ nodeId });
          console.log('[EdgeNet] Node created:', nodeId);
          edgeNetService.enableTimeCrystal(8);

          // Auto-start if consent was previously given
          const state = get();
          if (state.contributionSettings.consentGiven) {
            edgeNetService.startNode();
            set((s) => ({
              contributionSettings: { ...s.contributionSettings, enabled: true },
            }));
            console.log('[EdgeNet] Auto-started from previous session');
          }
        }
      }

      set({ isConnected: true, isLoading: false });
      console.log('[EdgeNet] Initialization complete');
    } catch (error) {
      console.error('[EdgeNet] Initialization failed:', error);
      set({
        error: error instanceof Error ? error.message : 'Failed to initialize',
        isLoading: false,
      });
    }
  },

  startContributing: () => {
    const { contributionSettings, isWASMReady } = get();
    if (!contributionSettings.consentGiven) {
      console.warn('[EdgeNet] Cannot start without consent');
      return;
    }
    if (isWASMReady) {
      edgeNetService.startNode();
      console.log('[EdgeNet] Started contributing');
    }
    set((state) => ({
      contributionSettings: { ...state.contributionSettings, enabled: true },
    }));
    get().saveToIndexedDB();
  },

  stopContributing: () => {
    edgeNetService.pauseNode();
    set((state) => ({
      contributionSettings: { ...state.contributionSettings, enabled: false },
    }));
    get().saveToIndexedDB();
    console.log('[EdgeNet] Stopped contributing');
  },

  updateRealStats: () => {
    const state = get();
    const sessionUptime = (Date.now() - state.startTime) / 1000;
    const totalUptime = sessionUptime + state.persistedUptime;
    const { isWASMReady, contributionSettings } = state;

    // Process epoch if contributing (advances WASM state)
    if (isWASMReady && contributionSettings.enabled) {
      edgeNetService.processEpoch();
      edgeNetService.stepCapabilities(1.0);
      edgeNetService.recordPerformance(0.95, 100);

      // Submit demo tasks periodically (every ~5 seconds) and process them
      if (Math.floor(sessionUptime) % 5 === 0) {
        edgeNetService.submitDemoTask();
      }
      // Process any queued tasks to earn credits
      edgeNetService.processNextTask().catch(() => {
        // No tasks available is normal
      });
    }

    // Get REAL stats from WASM node
    const realStats = edgeNetService.getStats();
    const timeCrystalSync = edgeNetService.getTimeCrystalSync();
    const networkFitness = edgeNetService.getNetworkFitness();

    // Debug: Log raw stats periodically
    if (realStats && Math.floor(sessionUptime) % 10 === 0) {
      console.log('[EdgeNet] Raw WASM stats:', {
        ruv_earned: realStats.ruv_earned?.toString(),
        tasks_completed: realStats.tasks_completed?.toString(),
        multiplier: realStats.multiplier,
        reputation: realStats.reputation,
        timeCrystalSync,
        networkFitness,
      });
    }

    if (realStats) {
      // Convert from nanoRuv (1e9) to Ruv
      const sessionRuvEarned = Number(realStats.ruv_earned) / 1e9;
      const sessionRuvSpent = Number(realStats.ruv_spent) / 1e9;
      const sessionTasks = Number(realStats.tasks_completed);

      // Add persisted values for cumulative totals
      const totalRuvEarned = state.persistedCredits + sessionRuvEarned;
      const totalTasks = state.persistedTasks + sessionTasks;

      set({
        stats: {
          totalNodes: contributionSettings.enabled ? 1 : 0,
          activeNodes: contributionSettings.enabled ? 1 : 0,
          totalCompute: Math.round(networkFitness * (contributionSettings.cpuLimit / 100) * 100) / 100,
          creditsEarned: Math.round(totalRuvEarned * 100) / 100,
          tasksCompleted: totalTasks,
          uptime: Math.round(totalUptime * 10) / 10,
          latency: Math.round((1 - timeCrystalSync) * 100),
          bandwidth: Math.round(contributionSettings.bandwidthLimit * 10) / 10,
        },
        timeCrystal: {
          ...state.timeCrystal,
          phase: (state.timeCrystal.phase + 0.01) % 1,
          coherence: Math.round(timeCrystalSync * 1000) / 1000,
          entropy: Math.round((1 - timeCrystalSync * 0.8) * 1000) / 1000,
          synchronizedNodes: contributionSettings.enabled ? 1 : 0,
        },
        credits: {
          available: Math.round((totalRuvEarned - sessionRuvSpent - state.credits.spent) * 100) / 100,
          pending: 0,
          earned: Math.round(totalRuvEarned * 100) / 100,
          spent: Math.round((sessionRuvSpent + state.credits.spent) * 100) / 100,
        },
        isConnected: isWASMReady,
        isLoading: false,
      });

      // Save to IndexedDB periodically (every 10 seconds worth of updates)
      if (Math.floor(sessionUptime) % 10 === 0) {
        get().saveToIndexedDB();
      }
    } else {
      // WASM not ready - show zeros but keep persisted values
      set({
        stats: {
          ...state.stats,
          totalNodes: 0,
          activeNodes: 0,
          totalCompute: 0,
          uptime: Math.round(totalUptime * 10) / 10,
          creditsEarned: state.persistedCredits,
          tasksCompleted: state.persistedTasks,
        },
        credits: {
          ...state.credits,
          earned: state.persistedCredits,
        },
        isConnected: false,
        isLoading: !isWASMReady,
      });
    }
  },
}));
