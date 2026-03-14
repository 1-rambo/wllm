/// <reference types="vite/client" />

// 让 TypeScript 认识 ?url 导入的 .wasm 文件
declare module '*.wasm?url' {
  const url: string;
  export default url;
}
