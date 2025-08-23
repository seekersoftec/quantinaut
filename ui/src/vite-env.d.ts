/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_WS_USER: string;
  readonly VITE_WS_PASS: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
