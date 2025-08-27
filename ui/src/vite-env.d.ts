/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_USER: string;
  readonly VITE_API_PASS: string;
  readonly VITE_API_HOST: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
