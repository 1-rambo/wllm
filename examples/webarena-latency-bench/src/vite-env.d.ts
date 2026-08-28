/// <reference types="vite/client" />

interface Window {
  wllmWebArenaReport?: import('./bench/types').BenchReport | null;
  wllmWebArenaError?: string | null;
}
